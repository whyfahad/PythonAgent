import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXTRACTION_URI = "ws://localhost:8001/extract"
SIMILARITY_AGENT_URI = "ws://localhost:8004/reason"
RELATION_AGENT_URI = "ws://localhost:8005/reason"
COORDINATOR_URI = "ws://localhost:8006/coordinator"

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
        goals = [target for rel, target in rels if rel in ["MotivatedByGoal", "Desires", "HasSubevent", "Causes", "CausesDesire", "UsedFor" ]]
        summary_data.append({
            "Concept": c,
            "Inferred Goals": ", ".join(goals) if goals else "None",
            "Relation Count": len(rels)
        })
        for rel, tgt in rels:
            rel_data.append({"Concept": c, "Relation": rel, "Target": tgt})

    concept_df = pd.DataFrame(summary_data)
    relation_df = pd.DataFrame(rel_data)

    print("\n📋 Extracted Concepts Summary:")
    print(concept_df.to_string(index=False))

    concept_df.plot(kind="bar", x="Concept", y="Relation Count", color="cornflowerblue", figsize=(10, 4), legend=False)
    plt.title("Number of ConceptNet Relations per Concept")
    plt.ylabel("Relation Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    if not relation_df.empty:
        print("\n🔗 ConceptNet Relations:")
        print(relation_df.to_string(index=False))

async def communicate_with_agents(user_input):
    # Step 1: Extract concepts
    async with websockets.connect(EXTRACTION_URI) as extract_ws:
        await extract_ws.send(user_input)
        raw = await extract_ws.recv()
        extract_response = json.loads(raw)

        print("\n[✓ Extraction Agent Response]")
        simplified = {k: v for k, v in extract_response.items() if k not in ['sentence_embedding', 'concept_embeddings']}
        print(json.dumps(simplified, indent=2))

        display_extracted_concept_visuals(extract_response)

    # Step 2: Round 1 - parallel
    async with websockets.connect(SIMILARITY_AGENT_URI) as sim_ws, websockets.connect(RELATION_AGENT_URI) as rel_ws:
        await asyncio.gather(
            sim_ws.send(json.dumps({"step": "round1", "input": extract_response})),
            rel_ws.send(json.dumps({"step": "round1", "input": extract_response}))
        )
        sim_result_1, rel_result_1 = await asyncio.gather(sim_ws.recv(), rel_ws.recv())
        sim_result_1 = json.loads(sim_result_1)
        rel_result_1 = json.loads(rel_result_1)

        print("\n[✓ Round 1 Completed]")
        print(f"Similarity Agent Top: {sim_result_1[0]['concept']} | Score: {sim_result_1[0]['score']}")
        print(f"Relation Agent Top:   {rel_result_1[0]['concept']} | Score: {rel_result_1[0]['score']}")
        plot_agent_scores("Similarity Agent", sim_result_1, "Round 1")
        plot_agent_scores("Relation Agent", rel_result_1, "Round 1")
        plot_combined_scores(sim_result_1, rel_result_1, "Round 1")

        # Step 3: Round 2 - cross-feedback
        await asyncio.gather(
            sim_ws.send(json.dumps({"step": "round2", "peer": rel_result_1})),
            rel_ws.send(json.dumps({"step": "round2", "peer": sim_result_1}))
        )
        sim_result_2, rel_result_2 = await asyncio.gather(sim_ws.recv(), rel_ws.recv())
        sim_result_2 = json.loads(sim_result_2)
        rel_result_2 = json.loads(rel_result_2)

        print("\n[✓ Round 2 Adjusted Scores]")
        print(f"Similarity Adjusted Top: {sim_result_2[0]['concept']} | Score: {sim_result_2[0]['score']}")
        print(f"Relation Adjusted Top:   {rel_result_2[0]['concept']} | Score: {rel_result_2[0]['score']}")
        plot_agent_scores("Similarity Agent", sim_result_2, "Round 2 (Adjusted)")
        plot_agent_scores("Relation Agent", rel_result_2, "Round 2 (Adjusted)")
        plot_combined_scores(sim_result_2, rel_result_2, "Round 2 (Adjusted)")

    # Step 4: Coordinator output
    async with websockets.connect(COORDINATOR_URI) as coord_ws:
        await coord_ws.send(json.dumps({
            "similarity": sim_result_2,
            "relation": rel_result_2
        }))
        final_response = json.loads(await coord_ws.recv())
        print("\n[🏁 Final Merged Inference from Coordinator]")
        for i, item in enumerate(final_response["final_inference"], 1):
            print(f"\n#{i}: Concept: {item['concept']}")
            print(f"   Composite Score: {item['composite_score']}")
            print(f"   Inferred Goals: {item['goals']}")
            print(f"   Supported by: {item['sources']}")

if __name__ == "__main__":
    user_input = input("Enter a user query: ")
    asyncio.run(communicate_with_agents(user_input))
