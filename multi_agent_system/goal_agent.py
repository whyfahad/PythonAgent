import redis
import json
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Redis config
r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "similarity_scored"
channel_publish = "goal_predicted"

# Use a lightweight model for now
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

goal_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=32,
    do_sample=False,
    device=-1  # Run on CPU
)

# Heuristics
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
    if re.match(r"^(to\s)?(be|get|do|have)$", goal_lower):
        return False
    return True

def predict_goals(concept_data):
    result = []

    for item in concept_data:
        concept = item["concept"]
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
            print(f"[GoalAgent]  Pipeline error for '{concept}': {e}")

        if not final_output and concept.lower() in RULE_BASED_GOALS:
            final_output = RULE_BASED_GOALS[concept.lower()]
            source = "Rule-based"

        result.append({
            "concept": concept,
            "goal": final_output or "",
            "source": source if final_output else "None"
        })

    return result

def run_goal_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[GoalAgent]  Subscribed to '{channel_subscribe}'")

    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        try:
            data = json.loads(message['data'].decode())
            print("[GoalAgent]  Received Input Concepts:")
            for item in data:
                print(f" - {item['concept']}")

            goal_results = predict_goals(data)

            print("[GoalAgent]  Publishing Inferred Goals...")
            for res in goal_results:
                print(f" - {res['concept']}: {res['goal']} (source: {res['source']})")

            r.publish(channel_publish, json.dumps(goal_results))
            print("[GoalAgent]  Published to", channel_publish)

        except Exception as e:
            print(f"[GoalAgent]  Error: {e}")

if __name__ == "__main__":
    run_goal_agent()
