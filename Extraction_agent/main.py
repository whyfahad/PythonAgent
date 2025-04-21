from fastapi import FastAPI, WebSocket
import uvicorn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.cocoex_utils import run_extraction_agent

app = FastAPI()

@app.websocket("/extract")
async def extract(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = run_extraction_agent(data)
            await websocket.send_json(result)
    except Exception as e:
        print(f"[Extraction Agent] Connection closed or error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8001, reload=True, timeout_keep_alive=120)