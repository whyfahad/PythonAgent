import redis
import json
import time

# Setup
r = redis.Redis(host='localhost', port=6379, db=0)
channel_publish = "user_input"
trivia_path = "triviaqa_sample.json"

# Load trivia questions
with open(trivia_path, "r") as f:
    trivia_questions = json.load(f)

# Publish each question one by one
for item in trivia_questions:
    question = item.get("question", "").strip()
    if question:
        print(f"[Feeder] Publishing question: {question}")
        r.publish(channel_publish, question)
        time.sleep(2)  # Give time for agents to process
