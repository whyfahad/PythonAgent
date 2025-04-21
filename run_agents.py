import asyncio
import websockets
import json

EXTRACTION_URI = "ws://localhost:8001/extract"
SIMILARITY_AGENT_URI = "ws://localhost:8004/reason"
RELATION_AGENT_URI = "ws://localhost:8005/reason"
COORDINATOR_URI = "ws://localhost:8006/coordinator"

async def communicate_with_agents(user_input):
    # Step 1: Send input to Extraction Agent
    async with websockets.connect(EXTRACTION_URI) as extract_ws:
        await extract_ws.send(user_input)
        extract_response = json.loads(await extract_ws.recv())

        # Display without embeddings
        display_response = dict(extract_response)
        display_response.pop("sentence_embedding", None)
        display_response.pop("concept_embeddings", None)

        print("\n[✓ Extraction Agent]")
        print(json.dumps(display_response, indent=2))

    # Step 2: Round 1 - Both agents perform initial reasoning
    async with websockets.connect(SIMILARITY_AGENT_URI) as sim_ws, websockets.connect(RELATION_AGENT_URI) as rel_ws:
        await sim_ws.send(json.dumps({
            "step": "round1",
            "input": extract_response
        }))
        await rel_ws.send(json.dumps({
            "step": "round1",
            "input": extract_response
        }))
        sim_round1 = json.loads(await sim_ws.recv())
        rel_round1 = json.loads(await rel_ws.recv())

        print("\n[✓ Round 1 Completed]")
        print(f"Similarity Agent Top: {sim_round1[0]['concept']} | Score: {sim_round1[0]['score']}")
        print(f"Relation Agent Top:   {rel_round1[0]['concept']} | Score: {rel_round1[0]['score']}")

        # Step 3: Round 2 - Agents receive peer suggestions
        await sim_ws.send(json.dumps({
            "step": "round2",
            "peer": rel_round1
        }))
        await rel_ws.send(json.dumps({
            "step": "round2",
            "peer": sim_round1
        }))
        sim_round2 = json.loads(await sim_ws.recv())
        rel_round2 = json.loads(await rel_ws.recv())

        print("\n[✓ Round 2 Adjusted Scores]")
        print(f"Similarity Adjusted Top: {sim_round2[0]['concept']} | Score: {sim_round2[0]['score']}")
        print(f"Relation Adjusted Top:   {rel_round2[0]['concept']} | Score: {rel_round2[0]['score']}")

    # Step 4: Send final adjusted results to Coordinator
    async with websockets.connect(COORDINATOR_URI) as coord_ws:
        await coord_ws.send(json.dumps({
            "similarity": sim_round2,
            "relation": rel_round2
        }))
        final = json.loads(await coord_ws.recv())

        print("\n[🏁 Final Merged Inference from Coordinator]")
        for i, item in enumerate(final["final_inference"], 1):
            print(f"\n#{i}: Concept: {item['concept']}")
            print(f"   Composite Score: {item['composite_score']}")
            print(f"   Inferred Goals: {item['goals']}")
            print(f"   Supported by: {item['sources']}")

if __name__ == "__main__":
    user_input = input("Enter a user query: ")
    asyncio.run(communicate_with_agents(user_input))
