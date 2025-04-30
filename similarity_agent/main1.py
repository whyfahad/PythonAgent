from fastapi import FastAPI, WebSocket
import uvicorn
import sys, os
import json

# Add parent path to access logic modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from similarity_agent.similarity_logic import (
    run_similarity_agent,
    adjust_similarity_scores_with_peer,
    generate_explanation
)

app = FastAPI()
agent_cache = {}

@app.websocket("/reason")
async def similarity_reasoning(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            step = message.get("step")

            if step == "round1":
                try:
                    extraction_data = message["input"]
                    sim_results = run_similarity_agent(extraction_data)
                    agent_cache[websocket] = (extraction_data, sim_results)
                    await websocket.send_json(sim_results)
                except Exception as e:
                    print(f"[Similarity Agent] Error in round1: {e}")
                    await websocket.close()
                    break

            elif step == "round2":
                try:
                    peer_summary = message["peer"]

                    if websocket not in agent_cache:
                        print("[Similarity Agent] No cached round1 data found.")
                        await websocket.close()
                        break

                    _, own_results = agent_cache[websocket]
                    adjusted_results = adjust_similarity_scores_with_peer(peer_summary, own_results)

                    for item in adjusted_results:
                        original = next((o for o in own_results if o["concept"] == item["concept"]), None)
                        if original:
                            old_score = original["original_score"]
                            new_score = item["score"]
                            item["confidence_delta"] = round(new_score - old_score, 4)
                            item["reason"] = generate_explanation(item["concept"], old_score, new_score)
                        else:
                            item["confidence_delta"] = 0.0
                            item["reason"] = "No previous score to compare."

                    await websocket.send_json(adjusted_results)

                except Exception as e:
                    print(f"[Similarity Agent] Error in round2: {e}")
                    await websocket.close()
                    break

    except Exception as e:
        print(f"[Similarity Agent] WebSocket closed or errored: {e}")

if __name__ == "__main__":
    uvicorn.run("similarity_agent.main1:app", host="localhost", port=8004, reload=True, timeout_keep_alive=120)
