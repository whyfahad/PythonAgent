from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
import uvicorn
import sys, os, json
import copy

# Extend path to access reasoning logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Reasoning_agent_relation.reasoning_relation import (
    run_reasoning_agent,
    adjust_relation_scores_with_peer,
    generate_explanation
)

app = FastAPI()
agent_cache = {}

@app.websocket("/reason")
async def relation_reasoning(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            step = message.get("step")

            if step == "round1":
                extraction_data = message["input"]
                rel_results = run_reasoning_agent(extraction_data)

                # Save both raw results and a snapshot of original scores
                score_snapshot = {entry["concept"]: entry["score"] for entry in rel_results}
                agent_cache[websocket] = (extraction_data, rel_results, score_snapshot)

                await websocket.send_json(rel_results)

            elif step == "round2":
                peer_summary = message["peer"]

                if websocket not in agent_cache:
                    print("[Relation Agent] No round1 cache found.")
                    break

                _, own_results, original_scores = agent_cache[websocket]
                adjusted_results = adjust_relation_scores_with_peer(peer_summary, own_results)

                for item in adjusted_results:
                    concept = item["concept"]
                    old_score = original_scores.get(concept, 0.0)
                    new_score = item["score"]

                    item["confidence_delta"] = round(new_score - old_score, 4)
                    item["reason"] = generate_explanation(concept, old_score, new_score)

                await websocket.send_json(adjusted_results)

            else:
                print("[Relation Agent] Unknown step received.")
                break

    except Exception as e:
        print(f"[Relation Agent] WebSocket closed or errored: {e}")

    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
        print("[Relation Agent] WebSocket session ended.")

if __name__ == "__main__":
    uvicorn.run("relation_agent.main2:app", host="localhost", port=8005, reload=True, timeout_keep_alive=120)
