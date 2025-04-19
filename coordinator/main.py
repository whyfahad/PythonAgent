from fastapi import FastAPI, WebSocket
import uvicorn
import json

app = FastAPI()

@app.websocket("/coordinator")
async def coordinator_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            similarity_results = message.get("similarity", [])
            relation_results = message.get("relation", [])

            concept_votes = {}

            # Collect and organize all results from both agents
            for agent_name, agent_data in [("SimilarityAgent", similarity_results), ("RelationAgent", relation_results)]:
                for item in agent_data:
                    concept = item["concept"]
                    score = item["score"]
                    goals = item["inferred_goals"]

                    if concept not in concept_votes:
                        concept_votes[concept] = []

                    concept_votes[concept].append({
                        "agent": agent_name,
                        "score": score,
                        "goals": goals
                    })

            # Compute final merged scores with weighted logic
            final_ranked = []
            for concept, votes in concept_votes.items():
                avg_score = sum(v["score"] for v in votes) / len(votes)
                all_goals = set(g for v in votes for g in v["goals"])
                goal_count = len(all_goals)
                composite_score = 0.8 * avg_score + 0.2 * goal_count

                final_ranked.append({
                    "concept": concept,
                    "avg_score": round(avg_score, 4),
                    "goal_count": goal_count,
                    "goals": list(all_goals),
                    "composite_score": round(composite_score, 4),
                    "sources": list({v["agent"] for v in votes})
                })

            # Sort by composite score
            final_ranked.sort(key=lambda x: x["composite_score"], reverse=True)

            await websocket.send_json({
                "final_inference": final_ranked[:3]
            })

    except Exception as e:
        print("[Coordinator] WebSocket closed or error:", str(e))

if __name__ == "__main__":
    uvicorn.run("coordinator.main:app", host="localhost", port=8006, reload=True)
