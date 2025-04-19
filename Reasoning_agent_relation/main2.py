from fastapi import FastAPI, WebSocket
import uvicorn
import sys, os

# Ensure parent directory is in the path to access reasoning_relation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Reasoning_agent_relation.reasoning_relation import run_reasoning_agent, adjust_relation_scores_with_peer

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
                agent_cache[websocket] = (extraction_data, rel_results)
                await websocket.send_json(rel_results)

            elif step == "round2":
                peer_summary = message["peer"]
                _, own_results = agent_cache.get(websocket, (None, []))
                adjusted_results = adjust_relation_scores_with_peer(peer_summary, own_results)
                await websocket.send_json(adjusted_results)

    except Exception as e:
        print(f"[Relation Agent] WebSocket closed or errored: {e}")

if __name__ == "__main__":
    uvicorn.run("relation_agent.main2:app", host="localhost", port=8005, reload=True)
