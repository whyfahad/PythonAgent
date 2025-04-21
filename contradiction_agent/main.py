# contradiction_agent/main.py
from fastapi import FastAPI, WebSocket
import uvicorn
import requests

app = FastAPI()

# Naive contradiction detection via ConceptNet Antonym relation
def get_antonyms(concept):
    url = f"http://api.conceptnet.io/c/en/{concept.replace(' ', '_')}"
    response = requests.get(url)
    antonyms = []
    if response.status_code == 200:
        edges = response.json().get("edges", [])
        for e in edges:
            if e["rel"]["label"] == "Antonym":
                antonyms.append(e["end"]["label"])
    return antonyms

@app.websocket("/check")
async def contradiction_check(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            concepts = data.get("concepts", [])
            contradiction_pairs = []
            for c1 in concepts:
                antonyms = get_antonyms(c1)
                for c2 in concepts:
                    if c2 != c1 and c2 in antonyms:
                        contradiction_pairs.append([c1, c2])
            await websocket.send_json({"contradictions": contradiction_pairs})
    except Exception as e:
        print(f"[ContradictionAgent] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=True)