import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pytest

EXTRACTION_URI = "ws://localhost:8001/extract"
SIMILARITY_AGENT_URI = "ws://localhost:8004/reason"
RELATION_AGENT_URI = "ws://localhost:8005/reason"
COORDINATOR_URI = "ws://localhost:8006/coordinator"
GOAL_URI = "ws://localhost:8007/predict"
CONTRADICTION_URI = "ws://localhost:8008/check"
RESPONSE_GEN_URI = "ws://localhost:8012/respond"  # New Response Generation Agent

# Load TriviaQA dataset
triviaqa_data = pd.read_json("triviaqa_sample.json")  # Assume small file for now
questions = triviaqa_data["question"]
answers = triviaqa_data["answer"]

@pytest.mark.asyncio
async def test_triviaqa():
    correct = 0
    total = len(questions)

    for idx, question in enumerate(questions):
        print(f"\n🚀 Processing Question {idx+1}/{total}: {question}")

        final_response = await communicate_with_agents(question)
        expected_answer = answers.iloc[idx].strip().lower()

        generated_answer = (final_response.get("generated_answer") or "").strip().lower()

        matched = False
        if expected_answer in generated_answer or generated_answer in expected_answer:
            matched = True

        if matched:
            print(f"✅ Matched! (Expected: {expected_answer})")
            correct += 1
        else:
            print(f"❌ Not Matched! (Expected: {expected_answer}), Got: {generated_answer}")

    accuracy = correct / total * 100
    print(f"\n🎯 TriviaQA Test Completed. Accuracy: {accuracy:.2f}%")

def plot_agent_scores(agent_name, results, round_label):
    df = pd.DataFrame({
        "Concept": [item["concept"] for item in results],
        "Score": [item["score"] for item in results]
    })
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Concept", y="Score", data=df, hue="Concept", dodge=False, palette="viridis")
    plt.title(f"{agent_name} - {round_label}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

def plot_combined_scores(sim_results, rel_results, round_label):
    sim_df = pd.DataFrame(sim_results)[["concept", "score"]].copy()
    sim_df["Agent"] = "Similarity Agent"
    rel_df = pd.DataFrame(rel_results)[["concept", "score"]].copy()
    rel_df["Agent"] = "Relation Agent"
    combined_df = pd.concat([sim_df, rel_df])
    plt.figure(figsize=(12, 5))
    sns.barplot(data=combined_df, x="concept", y="score", hue="Agent", palette="mako")
    plt.title(f"Combined Agent Scores - {round_label}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

def display_extracted_concept_visuals(extract_response):
    concepts = extract_response["concepts"]
    relations = extract_response.get("conceptnet_relations", {})

    summary_data = []
    rel_data = []

    for c in concepts:
        rels = relations.get(c, [])
        goals = [target for rel, target in rels if rel in ["MotivatedByGoal", "Desires", "HasSubevent", "Causes", "CausesDesire", "UsedFor"]]
        summary_data.append({
            "Concept": c,
            "Inferred Goals": ", ".join(goals) if goals else "None",
            "Relation Count": len(rels)
        })
        for rel, tgt in rels:
            rel_data.append({"Concept": c, "Relation": rel, "Target": tgt})

    concept_df = pd.DataFrame(summary_data)
    relation_df = pd.DataFrame(rel_data)

    print("\n[Extracted Concepts Summary]:")
    print(concept_df.to_string(index=False))

    concept_df.plot(kind="bar", x="Concept", y="Relation Count", color="cornflowerblue", figsize=(10, 4), legend=False)
    plt.title("Number of ConceptNet Relations per Concept")
    plt.ylabel("Relation Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    if not relation_df.empty:
        print("\n[ConceptNet Relations]:")
        print(relation_df.to_string(index=False))

async def communicate_with_agents(user_input):
    print("\n Sending input to Extraction Agent...")
    async with websockets.connect(EXTRACTION_URI) as extract_ws:
        await extract_ws.send(user_input)
        extract_response = json.loads(await extract_ws.recv())

    concepts = extract_response.get("concepts", [])

    print("\n Querying Goal and Contradiction Agents...")
    async with websockets.connect(GOAL_URI) as goal_ws:
        await goal_ws.send(json.dumps({"concepts": concepts}))
        goal_result = json.loads(await goal_ws.recv())

    async with websockets.connect(CONTRADICTION_URI) as contradiction_ws:
        await contradiction_ws.send(json.dumps({"concepts": concepts}))
        contradiction_result = json.loads(await contradiction_ws.recv())

    print("\n Starting Reasoning Round 1...")
    async with websockets.connect(SIMILARITY_AGENT_URI) as sim_ws, websockets.connect(RELATION_AGENT_URI) as rel_ws:
        await asyncio.gather(
            sim_ws.send(json.dumps({"step": "round1", "input": extract_response})),
            rel_ws.send(json.dumps({"step": "round1", "input": extract_response}))
        )
        sim_result_1, rel_result_1 = await asyncio.gather(sim_ws.recv(), rel_ws.recv())
        sim_result_1 = json.loads(sim_result_1)
        rel_result_1 = json.loads(rel_result_1)

        plot_combined_scores(sim_result_1, rel_result_1, "Round 1")

        print("\n Starting Reasoning Round 2...")
        await asyncio.gather(
            sim_ws.send(json.dumps({"step": "round2", "peer": rel_result_1})),
            rel_ws.send(json.dumps({"step": "round2", "peer": sim_result_1}))
        )
        sim_result_2, rel_result_2 = await asyncio.gather(sim_ws.recv(), rel_ws.recv())
        sim_result_2 = json.loads(sim_result_2)
        rel_result_2 = json.loads(rel_result_2)

        plot_combined_scores(sim_result_2, rel_result_2, "Round 2 (Adjusted)")

    print("\n Sending results to Coordinator...")
    async with websockets.connect(COORDINATOR_URI) as coord_ws:
        await coord_ws.send(json.dumps({
            "similarity": sim_result_2,
            "relation": rel_result_2,
            "goal_prediction": goal_result.get("predicted_goals", {}),
            "contradiction": contradiction_result.get("contradictions", [])
        }))
        final_response = json.loads(await coord_ws.recv())

    # ✅ Extract generated_answer
    generated_answer = final_response.get("generated_answer", "No answer generated.")
    print(f"\n🧠 Generated Answer: {generated_answer}")

    # --- Print results
    print("\n [Final Merged Inference from Coordinator]")
    for i, item in enumerate(final_response["final_inference"], 1):
        print(f"\n#{i}: Concept: {item['concept']}")
        print(f"   Composite Score: {item['composite_score']}")
        print(f"   Inferred Goals: {item['goals']}")
        print(f"   Goal Source: {item.get('goal_source', 'Unknown')}")
        print(f"   Supported by: {item['sources']}")
        print(f"   Confidence Delta: {item['confidence_delta']:.2f}")
        print(f"   Contradiction: {'Yes' if item.get('contradiction') else 'No'}")
        for reason in item.get("justifications", []):
            print(f"   Justification: {reason}")

    return final_response

if __name__ == "__main__":
    asyncio.run(test_triviaqa())
