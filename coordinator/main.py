from fastapi import FastAPI, WebSocket
import uvicorn
import json
import asyncio
import websockets

app = FastAPI()

GOAL_URI = "ws://localhost:8007/predict"
CONTRADICTION_URI = "ws://localhost:8008/check"

@app.websocket("/coordinator")
async def coordinator_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            similarity_results = message.get("similarity", [])
            relation_results = message.get("relation", [])

            all_concepts = list({item["concept"] for item in similarity_results + relation_results})

            # Request predictions from goal agent
            async with websockets.connect(GOAL_URI) as goal_ws:
                await goal_ws.send(json.dumps({"concepts": all_concepts}))
                goal_response = await goal_ws.recv()
                goal_data = json.loads(goal_response).get("predicted_goals", {})

            # Request contradictions from contradiction agent
            async with websockets.connect(CONTRADICTION_URI) as contradiction_ws:
                await contradiction_ws.send(json.dumps({"concepts": all_concepts}))
                contradiction_response = await contradiction_ws.recv()
                contradiction_data = json.loads(contradiction_response).get("contradictions", [])

            contradiction_set = {tuple(sorted(pair)) for pair in contradiction_data}

            concept_votes = {}

            # Collect and organize all results from both agents
            for agent_name, agent_data in [
                ("SimilarityAgent", similarity_results),
                ("RelationAgent", relation_results)
            ]:
                for item in agent_data:
                    concept = item["concept"]
                    score = item["score"]
                    goals = goal_data.get(concept, [])

                    if concept not in concept_votes:
                        concept_votes[concept] = []

                    concept_votes[concept].append({
                        "agent": agent_name,
                        "score": score,
                        "goals": goals
                    })

            # Compute final merged scores with goal boosts and contradiction penalties
            final_ranked = []
            for concept, votes in concept_votes.items():
                avg_score = sum(v["score"] for v in votes) / len(votes)
                all_goals = set(g for v in votes for g in v["goals"])
                goal_count = len(all_goals)
                has_contradiction = any(set([concept, other]) in contradiction_set for other in all_concepts if other != concept)

                composite_score = 0.7 * avg_score + 0.2 * goal_count - (0.1 if has_contradiction else 0)

                final_ranked.append({
                    "concept": concept,
                    "avg_score": round(avg_score, 4),
                    "goal_count": goal_count,
                    "goals": list(all_goals),
                    "contradiction": has_contradiction,
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
    uvicorn.run("main:app", host="localhost", port=8006, reload=True, timeout_keep_alive=120)
