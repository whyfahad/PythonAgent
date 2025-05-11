import redis
import json

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "final_answer"
channel_publish = "critic_feedback"

# === Critic Logic ===
def critique_concepts(final_ranking):
    feedback = []

    for item in final_ranking:
        concept = item.get("concept", "")
        justification = " ".join(item.get("justifications", []))
        confidence = item.get("score", 0.0)
        issues = []

        if len(justification.strip()) < 20:
            issues.append("Justification too short")

        if justification.lower().startswith(("probably", "maybe", "could")):
            issues.append("Justification lacks assertiveness")

        if confidence < 0.5:
            issues.append("Confidence score is low")

        if issues:
            feedback.append({
                "concept": concept,
                "confidence": confidence,
                "issues": issues,
                "recommendation": f"Review {concept}: {', '.join(issues)}"
            })

    return feedback

# === Redis Pub/Sub Loop ===
def run_critic_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[CriticAgent] Subscribed to '{channel_subscribe}'")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            data = json.loads(message["data"].decode())
            final_ranking = data.get("final_ranking", [])
            print("[CriticAgent] Received final_ranking.")

            feedback = critique_concepts(final_ranking)
            r.publish(channel_publish, json.dumps({"feedback": feedback}))
            print(f"[CriticAgent] Published {len(feedback)} critique notes.")

        except Exception as e:
            print(f"[CriticAgent] Error processing message: {str(e)}")

if __name__ == "__main__":
    run_critic_agent()
