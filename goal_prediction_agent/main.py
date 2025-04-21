from fastapi import FastAPI, WebSocket
import uvicorn
import spacy
import requests

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# Use ConceptNet to find goal-related relations
def get_goals_from_conceptnet(concept):
    url = f"http://api.conceptnet.io/c/en/{concept.replace(' ', '_')}"
    response = requests.get(url)
    goals = []
    if response.status_code == 200:
        edges = response.json().get("edges", [])
        for e in edges:
            rel = e["rel"]["label"]
            if rel in ["MotivatedByGoal", "CausesDesire"]:
                goals.append(e["end"]["label"])
    return goals

@app.websocket("/predict")
async def goal_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            concepts = data.get("concepts", [])
            predictions = {
                concept: get_goals_from_conceptnet(concept) for concept in concepts
            }
            await websocket.send_json({"predicted_goals": predictions})
    except Exception as e:
        print(f"[GoalPredictionAgent] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8007, reload=True)


