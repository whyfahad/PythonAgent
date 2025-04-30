from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
import uvicorn
import json

app = FastAPI()

@app.websocket("/debate")
async def debate_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_json()
                final_ranking = message.get("final_ranking", [])
                challenges = []

                for item in final_ranking:
                    concept = item["concept"]
                    goals = item.get("goals", [])
                    justification = " ".join(item.get("justifications", []))
                    confidence_delta = item.get("confidence_delta", 0.0)
                    sources = item.get("sources", [])

                    # Challenge criteria
                    weak_justification = len(justification.strip()) < 15
                    missing_goal = not goals
                    low_confidence = confidence_delta < 0.02
                    weak_support = len(sources) < 2

                    if weak_justification or missing_goal or low_confidence or weak_support:
                        challenges.append({
                            "concept": concept,
                            "issues": {
                                "missing_goal": missing_goal,
                                "low_confidence": low_confidence,
                                "weak_justification": weak_justification,
                                "weak_support": weak_support
                            },
                            "comment": f"{concept} flagged for review. Justification: {justification[:60]}..."
                        })

                await websocket.send_json({"challenges": challenges})

            except Exception as msg_err:
                print("[DebaterAgent] Message handling error:", str(msg_err))
                break

    except Exception as e:
        print("[DebaterAgent] WebSocket error:", str(e))

    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("[DebaterAgent] WebSocket session ended.")

if __name__ == "__main__":
    uvicorn.run("debater_agent:app", host="localhost", port=8011, reload=True, timeout_keep_alive=120)
