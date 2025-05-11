import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "final_answer"
channel_publish = "debater_challenges"

def analyze_concepts(final_ranking):
    challenges = []

    for item in final_ranking:
        concept = item.get("concept", "")
        goals = item.get("goals", [])
        justification = " ".join(item.get("justifications", [])) if item.get("justifications") else ""
        confidence_delta = item.get("confidence_delta", 0.0)
        sources = item.get("sources", [])

        weak_justification = len(justification.strip()) < 15
        missing_goal = not goals or not goals[0] or len(goals[0].strip()) < 5
        low_confidence = confidence_delta < 0.02
        weak_support = len(sources) < 2

        if weak_justification or missing_goal or low_confidence or weak_support:
            issues = {
                "missing_goal": missing_goal,
                "low_confidence": low_confidence,
                "weak_justification": weak_justification,
                "weak_support": weak_support
            }
            comment_parts = []
            if missing_goal: comment_parts.append("Missing or weak goal")
            if low_confidence: comment_parts.append("Low confidence delta")
            if weak_justification: comment_parts.append("Short/unclear justification")
            if weak_support: comment_parts.append("Only one agent supported")

            challenges.append({
                "concept": concept,
                "issues": issues,
                "comment": f"Issues with '{concept}': {', '.join(comment_parts)}. Justification: '{justification[:60]}...'"
            })

    return challenges

def run_debater_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[DebaterAgent]  Subscribed to '{channel_subscribe}'")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            data = json.loads(message["data"].decode())
            final_ranking = data.get("final_ranking", [])
            if not final_ranking:
                print("[DebaterAgent]  No final_ranking in received data.")
                continue

            print(f"[DebaterAgent]  Evaluating {len(final_ranking)} concept(s)...")
            challenges = analyze_concepts(final_ranking)

            r.publish(channel_publish, json.dumps({"challenges": challenges}))

            print(f"[DebaterAgent]  Published {len(challenges)} challenge(s).")
            for ch in challenges:
                print(" -", ch["comment"])

        except Exception as e:
            print("[DebaterAgent]  Error:", str(e))

if __name__ == "__main__":
    run_debater_agent()
