from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
import uvicorn
import json
import asyncio
import websockets

app = FastAPI()

GOAL_URI = "ws://localhost:8007/predict"
CONTRADICTION_URI = "ws://localhost:8008/check"
CRITIC_URI = "ws://localhost:8009/critic"
DEBATER_URI = "ws://localhost:8011/debate"
VERIFIER_URI = "ws://localhost:8010/verify"
RESPONSE_GENERATION_URI = "ws://localhost:8012/generate"

@app.websocket("/coordinator")
async def coordinator_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_json()
                similarity_results = message.get("similarity", [])
                relation_results = message.get("relation", [])
                all_concepts = list({item["concept"] for item in similarity_results + relation_results})

                # --- Goal Prediction Agent ---
                try:
                    async with websockets.connect(GOAL_URI) as goal_ws:
                        await goal_ws.send(json.dumps({"concepts": all_concepts}))
                        goal_response = await goal_ws.recv()
                        goal_data = json.loads(goal_response).get("predicted_goals", {})
                except Exception as e:
                    print("[Coordinator] Goal Agent error:", str(e))
                    goal_data = {}

                # --- Contradiction Agent ---
                try:
                    async with websockets.connect(CONTRADICTION_URI) as contradiction_ws:
                        await contradiction_ws.send(json.dumps({"concepts": all_concepts}))
                        contradiction_response = await contradiction_ws.recv()
                        contradiction_data = json.loads(contradiction_response).get("contradictions", [])
                except Exception as e:
                    print("[Coordinator] Contradiction Agent error:", str(e))
                    contradiction_data = []

                contradiction_set = {tuple(sorted(pair)) for pair in contradiction_data}
                concept_votes = {}

                # --- Merge Reasoning Agent Results ---
                for agent_name, agent_data in [("SimilarityAgent", similarity_results), ("RelationAgent", relation_results)]:
                    for item in agent_data:
                        concept = item["concept"]
                        score = item["score"]
                        reason = item.get("reason") or item.get("explanation") or item.get("inference") or "No justification provided."
                        confidence_delta = item.get("confidence_delta", 0.0)
                        predicted_goals = goal_data.get(concept, [])

                        if concept not in concept_votes:
                            concept_votes[concept] = []

                        concept_votes[concept].append({
                            "agent": agent_name,
                            "score": score,
                            "goals": predicted_goals,
                            "reason": reason,
                            "confidence_delta": confidence_delta
                        })

                # --- Final Ranking Computation ---
                final_ranked = []
                for concept, votes in concept_votes.items():
                    avg_score = sum(v["score"] for v in votes) / len(votes)
                    total_delta = sum(v["confidence_delta"] for v in votes)
                    normalized_delta = total_delta / len(votes)

                    goal_entry = goal_data.get(concept, {})
                    goal_text = goal_entry.get("goal", "") if isinstance(goal_entry, dict) else goal_entry
                    goal_source = goal_entry.get("source", "Unknown") if isinstance(goal_entry, dict) else "Unknown"
                    goal_boost = 0.1 if goal_text and len(goal_text.strip()) > 4 and "something" not in goal_text.lower() else 0.0

                    has_contradiction = any(set([concept, other]) in contradiction_set for other in all_concepts if other != concept)

                    composite_score = round(
                        0.6 * avg_score +
                        0.2 * normalized_delta +
                        goal_boost -
                        (0.1 if has_contradiction else 0),
                        4
                    )

                    final_ranked.append({
                        "concept": concept,
                        "avg_score": round(avg_score, 4),
                        "goal_count": 1 if goal_text else 0,
                        "goals": [goal_text] if goal_text else [],
                        "goal_source": goal_source,
                        "contradiction": has_contradiction,
                        "composite_score": composite_score,
                        "sources": list({v["agent"] for v in votes}),
                        "confidence_delta": round(total_delta, 4),
                        "justifications": [v["reason"] for v in votes]
                    })

                final_ranked.sort(key=lambda x: x["composite_score"], reverse=True)

                # --- Critic Agent ---
                try:
                    async with websockets.connect(CRITIC_URI) as critic_ws:
                        await critic_ws.send(json.dumps({"final_ranking": final_ranked}))
                        critic_feedback = json.loads(await critic_ws.recv())
                        penalty_map = {f["concept"]: f["penalty"] for f in critic_feedback.get("feedback", [])}

                        for item in final_ranked:
                            item["composite_score"] = round(item["composite_score"] + penalty_map.get(item["concept"], 0.0), 4)

                        final_ranked.sort(key=lambda x: x["composite_score"], reverse=True)
                except Exception as ce:
                    print("[Coordinator] Critic agent unavailable:", str(ce))

                # --- Debater Agent ---
                debater_feedback = {}
                try:
                    async with websockets.connect(DEBATER_URI) as debater_ws:
                        await debater_ws.send(json.dumps({"final_ranking": final_ranked}))
                        debater_response = json.loads(await debater_ws.recv())
                        print("[Debater Feedback]:", json.dumps(debater_response, indent=2))
                        debater_feedback = debater_response.get("challenges", [])
                except Exception as de:
                    print("[Coordinator] Debater agent unavailable:", str(de))

                # --- Verifier Agent ---
                verifier_feedback = {}
                try:
                    async with websockets.connect(VERIFIER_URI) as verifier_ws:
                        await verifier_ws.send(json.dumps({"final_ranking": final_ranked}))
                        verifier_response = json.loads(await verifier_ws.recv())
                        print("[Verifier Feedback]:", json.dumps(verifier_response, indent=2))
                        verifier_feedback = verifier_response.get("challenges", [])
                except Exception as ve:
                    print("[Coordinator] Verifier agent unavailable:", str(ve))

                # --- Response Generation Agent (NEW) ---
                generated_answer = ""
                try:
                    top_concepts = [item["concept"] for item in final_ranked[:3]]  # Top 3 concepts
                    async with websockets.connect(RESPONSE_GENERATION_URI) as response_ws:
                        await response_ws.send(json.dumps({"concepts": top_concepts}))
                        response_data = await response_ws.recv()
                        response_obj = json.loads(response_data)
                        generated_answer = response_obj.get("generated_answer", "")
                        print("[ResponseGeneration] Generated Answer:", generated_answer)
                except Exception as re:
                    print("[Coordinator] Response generation agent unavailable:", str(re))

                # --- Final Output (updated) ---
                await websocket.send_json({
                    "final_inference": final_ranked[:3],
                    "debater_feedback": debater_feedback,
                    "verifier_feedback": verifier_feedback,
                    "generated_answer": generated_answer   # <-- Add to output
                })

            except Exception as msg_err:
                print("[Coordinator] Message handling error:", str(msg_err))
                break

    except Exception as outer:
        print("[Coordinator] WebSocket outer error:", str(outer))

    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("[Coordinator] WebSocket session ended.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8006, reload=True, timeout_keep_alive=120)
