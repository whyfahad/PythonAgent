from fastapi import FastAPI, WebSocket
import uvicorn
import sys, os

# Add parent path to access logic modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from similarity_agent.similarity_logic import run_similarity_agent, adjust_similarity_scores_with_peer

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
                extraction_data = message["input"]
                sim_results = run_similarity_agent(extraction_data)
                agent_cache[websocket] = (extraction_data, sim_results)
                await websocket.send_json(sim_results)

            elif step == "round2":
                peer_summary = message["peer"]
                _, own_results = agent_cache.get(websocket, (None, []))
                adjusted_results = adjust_similarity_scores_with_peer(peer_summary, own_results)
                await websocket.send_json(adjusted_results)

    except Exception as e:
        print(f"[Similarity Agent] WebSocket closed or errored: {e}")

if __name__ == "__main__":
    uvicorn.run("similarity_agent.main1:app", host="localhost", port=8004, reload=True, timeout_keep_alive=120)
