import redis
import json
import time
from itertools import combinations

r = redis.Redis(host='localhost', port=6379, db=0)

sub_channels = [
    "similarity_adjusted",
    "relation_adjusted",
    "goal_predicted",
    "debater_challenges",
    "verification_result"
]

publish_channel = "coordinator_output"

data_cache = {
    "similarity": None,
    "relation": None,
    "goal_predicted": None,
    "debater": None,
    "verifier": None,
    "round2_triggered": False,
    "timestamp": time.time()
}

def merge_and_rank():
    sim = data_cache["similarity"] or []
    rel = data_cache["relation"] or []
    goals = data_cache["goal_predicted"] or {}
    debater_feedback = data_cache["debater"] or []
    verifier_feedback = data_cache["verifier"] or []

    all_concepts = list({item["concept"] for item in sim + rel})
    concept_votes = {}

    for agent_name, dataset in [("SimilarityAgent", sim), ("RelationAgent", rel)]:
        for item in dataset:
            concept = item["concept"]
            concept_votes.setdefault(concept, []).append({
                "agent": agent_name,
                "score": item["score"],
                "goals": goals.get(concept, []),
                "reason": item.get("reason") or item.get("explanation", "N/A"),
                "confidence_delta": item.get("confidence_delta", 0.0)
            })

    final_ranked = []
    for concept, votes in concept_votes.items():
        avg_score = sum(v["score"] for v in votes) / len(votes)
        delta = sum(v["confidence_delta"] for v in votes)
        normalized_delta = delta / len(votes)

        goal_raw = goals.get(concept, "")
        if isinstance(goal_raw, list):
            goal_text = goal_raw[0] if goal_raw else ""
        elif isinstance(goal_raw, dict):
            goal_text = goal_raw.get("goal", "")
        else:
            goal_text = goal_raw

        goal_boost = 0.1 if goal_text and len(goal_text.strip()) > 4 else 0.0

        composite_score = round(
            0.6 * avg_score + 0.2 * normalized_delta + goal_boost,
            4
        )

        final_ranked.append({
            "concept": concept,
            "avg_score": round(avg_score, 4),
            "goals": [goal_text] if goal_text else [],
            "composite_score": composite_score,
            "contradiction": False,
            "confidence_delta": round(delta, 4),
            "sources": list({v["agent"] for v in votes}),
            "justifications": [v["reason"] for v in votes]
        })

    final_ranked.sort(key=lambda x: x["composite_score"], reverse=True)

    result = {
        "extracted_concepts": list({item["concept"] for item in sim + rel}),
        "similarity_results": sim,
        "relation_results": rel,
        "goal_predictions": goals,
        "critic_feedback": {},
        "debater_feedback": debater_feedback,
        "verifier_feedback": verifier_feedback,
        "final_ranking": final_ranked[:3],
        "answer": final_ranked[0]["concept"] if final_ranked else "No valid concepts",
        "concepts_used": [item["concept"] for item in final_ranked[:3]]
    }

    print("\n PUBLISHING FINAL RESULT:\n", json.dumps(result, indent=2), flush=True)
    r.publish(publish_channel, json.dumps(result))
    print("[Coordinator]  Published final merged result.", flush=True)

def run_coordinator():
    pubsub = r.pubsub()
    pubsub.subscribe(sub_channels)
    print("[Coordinator]  Subscribed to:", sub_channels, flush=True)

    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue

        try:
            channel = msg["channel"].decode()
            payload = json.loads(msg["data"].decode())

            if channel == "similarity_adjusted":
                data_cache["similarity"] = payload
            elif channel == "relation_adjusted":
                data_cache["relation"] = payload
            elif channel == "goal_predicted":
                goal_map = {}
                for item in payload:
                    goal_map[item["concept"]] = item.get("goal", item.get("inferred_goals", ""))
                data_cache["goal_predicted"] = goal_map

                # Trigger Round 2 if needed
                if data_cache["similarity"] is None and r.get("similarity_last_results"):
                    data_cache["similarity"] = json.loads(r.get("similarity_last_results"))
                if data_cache["relation"] is None and r.get("relation_last_results"):
                    data_cache["relation"] = json.loads(r.get("relation_last_results"))

                if data_cache["similarity"] and data_cache["relation"] and not data_cache["round2_triggered"]:
                    print("[Coordinator]  Triggering peer feedback for Round 2...", flush=True)
                    r.publish("peer_similarity", json.dumps(data_cache["relation"]))
                    r.publish("peer_relation", json.dumps(data_cache["similarity"]))
                    print("[Coordinator]  Sent peer_similarity and peer_relation", flush=True)
                    data_cache["round2_triggered"] = True

            elif channel == "debater_challenges":
                data_cache["debater"] = payload.get("challenges", [])
            elif channel == "verification_result":
                data_cache["verifier"] = payload.get("challenges", [])

            if data_cache["similarity"] and data_cache["relation"] and data_cache["goal_predicted"]:
                merge_and_rank()
                for k in ["similarity", "relation", "goal_predicted"]:
                    data_cache[k] = None
                data_cache["round2_triggered"] = False

        except Exception as e:
            print("[Coordinator]  Error:", str(e), flush=True)

if __name__ == "__main__":
    run_coordinator()
