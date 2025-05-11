import redis
import json
from transformers import pipeline

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "coordinator_output"
channel_publish = "verification_result"

# Load Natural Language Inference pipeline
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

def check_entailment(concept, goal_text):
    try:
        result = nli(f"{concept}. {goal_text}", truncation=True)[0]
        return result['label'], result['score']
    except Exception as e:
        return "ERROR", 0.0

def verify_inference(data):
    final_inference = data.get("final_ranking", [])
    challenges = []

    for item in final_inference:
        concept = item.get("concept", "")
        sources = item.get("sources", [])
        confidence_delta = item.get("confidence_delta", 0.0)
        justifications = item.get("justifications", [])
        justification = " ".join(justifications).strip() if justifications else ""
        goals = item.get("goals", [])

        # 1. Multi-agent agreement
        if "SimilarityAgent" in sources and "RelationAgent" in sources:
            pass  # ✅ Acceptable: support from both agents
        else:
            challenges.append({
                "concept": concept,
                "issue": "Only one agent supported.",
                "comment": f"{concept} support insufficient. Sources: {sources}"
            })

        # 2. Delta too small but final score is low
        if abs(confidence_delta) < 0.01 and item.get("composite_score", 0) < 0.4:
            challenges.append({
                "concept": concept,
                "issue": "No meaningful confidence adjustment.",
                "comment": f"{concept} had low delta={confidence_delta:.4f}. Justification: {justification[:60]}..."
            })

        # 3. Goal checks
        goal_text = goals[0].strip() if goals and isinstance(goals[0], str) else ""
        if len(goal_text) < 5:
            challenges.append({
                "concept": concept,
                "issue": "Missing or weak goal.",
                "comment": f"{concept} has weak or empty goal: '{goal_text}'"
            })
        else:
            label, score = check_entailment(concept, goal_text)
            if label == "ENTAILMENT" and score >= 0.8:
                pass  # ✅ Entailment is strong
            else:
                challenges.append({
                    "concept": concept,
                    "issue": "Goal not logically entailed.",
                    "comment": f"Entailment check failed: {label} ({score:.2f})"
                })

    return challenges


def run_verifier_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[VerifierAgent]  Subscribed to '{channel_subscribe}'")

    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        try:
            data = json.loads(message['data'].decode())
            print("[VerifierAgent]  Received final output from Coordinator.")

            challenges = verify_inference(data)
            payload = {"challenges": challenges}
            r.publish(channel_publish, json.dumps(payload))

            print(f"[VerifierAgent]  Published {len(challenges)} verification result(s).")
            for ch in challenges:
                print(f" -  {ch['concept']} → {ch['issue']} | {ch['comment']}")

        except Exception as e:
            print("[VerifierAgent]  Error:", str(e))

if __name__ == "__main__":
    run_verifier_agent()
