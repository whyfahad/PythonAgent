import redis
import json
from datetime import datetime

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)
pubsub = r.pubsub()
pubsub.subscribe("final_answer", "coordinator_output", "verification_result")

print("🔎 Listening for complete multi-agent output...\n")

def print_separator():
    print("\n" + "-" * 80 + "\n")

for msg in pubsub.listen():
    if msg["type"] != "message":
        continue

    try:
        data = json.loads(msg["data"].decode())
    except Exception as e:
        print("❌ Failed to decode JSON:", e)
        continue

    channel = msg["channel"].decode()
    timestamp = datetime.now().strftime("%H:%M:%S")

    if channel == "final_answer":
        print_separator()
        print(f"🗣️ [{timestamp}] FINAL ANSWER (from ResponseAgent):")
        print(json.dumps(data, indent=2))

    elif channel == "coordinator_output":
        print_separator()
        print(f"📦 [{timestamp}] FULL MULTI-AGENT RESULT (from Coordinator):")
        print(json.dumps(data, indent=2))

    elif channel == "verification_result":
        print_separator()
        print(f"🔍 [{timestamp}] VERIFIER FEEDBACK (from VerifierAgent):")
        if "challenges" in data and data["challenges"]:
            for ch in data["challenges"]:
                print(f"⚠️  Issue with concept '{ch['concept']}': {ch['issue']}")
                print(f"   ↪ {ch['comment']}")
        else:
            print("✅ No verification issues found.")
