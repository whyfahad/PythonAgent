from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
import torch
import json
import re
import uvicorn

app = FastAPI()

# === Model Configuration ===
model_name = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

goal_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=32,
    do_sample=False,
)

# === Fallback and Filtering Configuration ===
DISCARD_KEYWORDS = ["something", "get a job", "be a good friend", "do something"]
MIN_GOAL_LEN = 6
RULE_BASED_GOALS = {
    "thirsty": "to drink water",
    "hungry": "to eat food",
    "tired": "to rest or sleep",
    "angry": "to calm down",
    "confused": "to understand something",
    "lost": "to find direction or help"
}

def is_valid_goal(goal: str) -> bool:
    goal_lower = goal.lower()
    if any(kw in goal_lower for kw in DISCARD_KEYWORDS):
        return False
    if len(goal.strip()) < MIN_GOAL_LEN:
        return False
    if re.match(r"^(to\\s)?(be|get|do|have)$", goal_lower):
        return False
    return True

# === WebSocket Endpoint ===
@app.websocket("/predict")
async def predict_goals(websocket: WebSocket):
    await websocket.accept()
    print("[GoalPredictionAgent] WebSocket connection accepted.")
    try:
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                concepts = data.get("concepts", [])
                response = {}

                for concept in concepts:
                    prompt = f"What is the most likely goal of someone who is '{concept}'?"
                    fallback_prompt = f"What would someone who is '{concept}' most likely want to do?"
                    final_output = ""
                    source = "LLM"

                    try:
                        output = goal_generator(prompt)[0]["generated_text"].strip()
                        if is_valid_goal(output):
                            final_output = output
                        else:
                            fallback_output = goal_generator(fallback_prompt)[0]["generated_text"].strip()
                            if is_valid_goal(fallback_output):
                                final_output = fallback_output
                    except Exception as e:
                        print(f"[GoalPredictionAgent] Pipeline error for '{concept}': {e}")

                    if not final_output and concept.lower() in RULE_BASED_GOALS:
                        final_output = RULE_BASED_GOALS[concept.lower()]
                        source = "Rule-based"

                    response[concept] = {
                        "goal": final_output or "",
                        "source": source if final_output else "None"
                    }

                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps({"predicted_goals": response}))
                else:
                    print("[GoalPredictionAgent] WebSocket was not connected at send time.")
                    break

            except Exception as inner_error:
                print(f"[GoalPredictionAgent] Message error: {inner_error}")
                break

    except Exception as outer_error:
        print(f"[GoalPredictionAgent] WebSocket session error: {outer_error}")

    finally:
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except RuntimeError:
            pass
        print("[GoalPredictionAgent] WebSocket session ended.")

# === Server Runner ===
if __name__ == "__main__":
    uvicorn.run("goal_prediction_agent.main:app", host="localhost", port=8007, reload=True, timeout_keep_alive=120)
