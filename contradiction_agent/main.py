from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState  # ✅ Add this
import uvicorn
import json
from itertools import combinations

app = FastAPI()

@app.websocket("/check")
async def contradiction_checker(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                concepts = data.get("concepts", [])
                conceptnet_relations = data.get("conceptnet_relations", {})  # 🔥 must be passed from extractor

                contradictions = []
                for c1, c2 in combinations(concepts, 2):
                    if are_contradictory(c1, c2, conceptnet_relations):
                        contradictions.append([c1, c2])

                await websocket.send_json({"contradictions": contradictions})
            except Exception as e:
                print("[ContradictionAgent] Error:", str(e))
                break
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("[ContradictionAgent] WebSocket session ended.")

def are_contradictory(concept1, concept2, conceptnet_relations):
    c1_rels = conceptnet_relations.get(concept1.lower(), [])
    c2_rels = conceptnet_relations.get(concept2.lower(), [])

    # Check if either concept is listed as an 'Antonym' of the other
    antonyms_c1 = {target.lower() for rel, target in c1_rels if rel == "Antonym"}
    antonyms_c2 = {target.lower() for rel, target in c2_rels if rel == "Antonym"}

    return concept2.lower() in antonyms_c1 or concept1.lower() in antonyms_c2

if __name__ == "__main__":
    uvicorn.run("contradiction_agent.main:app", host="localhost", port=8008, reload=True, timeout_keep_alive=120)
