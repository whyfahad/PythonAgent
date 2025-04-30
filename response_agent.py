from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
import uvicorn
import requests
import json

# 🔥 Your normal OpenAI API Key
API_KEY = "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"

# 🔥 Correct API URL for normal chat (NOT realtime!)
API_URL = "https://api.openai.com/v1/chat/completions"

# 🔥 Model to use
MODEL = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo"

app = FastAPI()

@app.websocket("/generate")
async def response_generation_agent(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_json()
                concepts = message.get("concepts", [])

                if not concepts:
                    await websocket.send_json({"generated_answer": "No concepts provided."})
                    continue

                prompt = (
                    f"Given the following important concepts extracted from a question: {', '.join(concepts)}.\n"
                    f"Based on these concepts, infer what the original question might be asking about and generate a short and accurate answer.\n"
                    f"Only provide the answer, without repeating the concepts or the question."
                )

                # --- Normal OpenAI ChatCompletion request ---
                response = requests.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI answering trivia questions."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 50
                    }
                )

                data = response.json()

                if "choices" in data:
                    generated_answer = data["choices"][0]["message"]["content"].strip()
                else:
                    generated_answer = "Error generating answer."

                await websocket.send_json({"generated_answer": generated_answer})

            except Exception as e:
                print("[ResponseGenerationAgent] Error:", str(e))
                if websocket.application_state != WebSocketState.DISCONNECTED:
                    await websocket.send_json({"generated_answer": "Error generating answer."})
                break

    except Exception as outer:
        print("[ResponseGenerationAgent] Outer error:", str(outer))

    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("[ResponseGenerationAgent] WebSocket session ended.")

if __name__ == "__main__":
    uvicorn.run("response_agent:app", host="localhost", port=8012, reload=True, timeout_keep_alive=120)
