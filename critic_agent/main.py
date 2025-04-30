from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState  # ✅ Needed for safe close
import uvicorn
import json

app = FastAPI()

@app.websocket("/critic")
async def critic_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_json()
                final_ranking = message.get("final_ranking", [])

                feedback = []
                for concept_data in final_ranking:
                    concept = concept_data["concept"]
                    goals = concept_data.get("goals", [])
                    delta = concept_data.get("confidence_delta", 0.0)
                    justifications = concept_data.get("justifications", [])
                    contradiction = concept_data.get("contradiction", False)

                    # Critique conditions
                    missing_goals = len(goals) == 0
                    weak_goals = [g for g in goals if "something" in g.lower() or len(g.strip()) < 5]
                    weak_justification = any("no justification" in j.lower() or len(j.strip()) < 10 for j in justifications)
                    low_confidence = abs(delta) < 0.05

                    score_penalty = 0
                    if missing_goals:
                        score_penalty -= 0.1
                    if weak_goals:
                        score_penalty -= 0.05
                    if contradiction:
                        score_penalty -= 0.1
                    if weak_justification:
                        score_penalty -= 0.05
                    if low_confidence:
                        score_penalty -= 0.02

                    feedback.append({
                        "concept": concept,
                        "penalty": round(score_penalty, 4),
                        "flags": {
                            "missing_goals": missing_goals,
                            "weak_goals": weak_goals,
                            "contradiction": contradiction,
                            "low_confidence_delta": low_confidence,
                            "weak_justification": weak_justification
                        }
                    })

                # Log to terminal
                print("\n[🧠 Critic Agent Feedback]")
                for item in feedback:
                    print(f"\n- {item['concept']}: Penalty {item['penalty']}")
                    for k, v in item["flags"].items():
                        if v:
                            print(f"  ⚠️ {k.replace('_', ' ').capitalize()}: {v}")

                await websocket.send_json({
                    "status": "Critique completed",
                    "feedback": feedback
                })

            except Exception as e:
                print("[CriticAgent] Message handling error:", str(e))
                break

    except Exception as e:
        print("[CriticAgent] WebSocket connection error:", str(e))


    finally:

        try:

            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()

        except RuntimeError:

            pass  # WebSocket already closed

        print("[CriticAgent] WebSocket session ended.")

if __name__ == "__main__":
    uvicorn.run("critic_agent.main:app", host="localhost", port=8009, reload=True, timeout_keep_alive=120)
