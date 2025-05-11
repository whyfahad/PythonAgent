import redis
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)

# Redis channels
CHANNELS = {
    "subscribe_round1": "concept_extracted",
    "publish_round1": "relation_scored",
    "subscribe_round2": "peer_relation",
    "publish_round2": "relation_adjusted"
}

# Goal-related ConceptNet relations
GOAL_RELS = {"MotivatedByGoal", "Desires", "UsedFor", "Causes", "CausesDesire", "HasSubevent"}

# Round 1 Reasoning Logic
def compute_initial_scores(extraction):
    concepts = extraction["concepts"]
    embeddings = np.array(extraction["concept_embeddings"])
    sentence_emb = np.array(extraction["sentence_embedding"])
    relations = extraction["conceptnet_relations"]
    llm_goals = extraction.get("inferred_goals", {})

    results = []
    max_rel_count = max((len(r) for r in relations.values()), default=1)

    for i, concept in enumerate(concepts):
        sim = cosine_similarity([sentence_emb], [embeddings[i]])[0][0]
        rels = relations.get(concept, [])

        conceptnet_goals = [target for rel, target in rels if rel in GOAL_RELS]
        goals = conceptnet_goals if conceptnet_goals else llm_goals.get(concept, [])
        goal_source = "ConceptNet" if conceptnet_goals else "LLM (fallback)"

        score = 0.3 * sim + 0.7 * len(rels) / max_rel_count

        results.append({
            "agent": "RelationAgent",
            "concept": concept,
            "score": round(score, 4),
            "original_score": round(score, 4),
            "inferred_goals": goals,
            "relation_count": len(rels),
            "explanation": f"Score based on similarity ({round(sim, 2)}) and {len(rels)} ConceptNet relations. Goal source: {goal_source}.",
            "confidence_delta": 0.0
        })

    return results

# Round 2 Adjustment Logic
def adjust_scores_with_peer_feedback(peer_results, own_results):
    peer_map = {item["concept"]: set(item["inferred_goals"]) for item in peer_results}

    for item in own_results:
        concept = item["concept"]
        own_goals = set(item["inferred_goals"])
        peer_goals = peer_map.get(concept, set())
        original = item["original_score"]

        explanation_suffix = ""
        new_score = original

        if own_goals & peer_goals:
            new_score *= 1.1
            explanation_suffix = " Boosted due to goal agreement with peer."
        elif not own_goals and not peer_goals:
            new_score *= 0.8
            explanation_suffix = " Penalized due to lack of inferred goals by both agents."
        else:
            explanation_suffix = " No consensus on inferred goals."

        delta = round(new_score - original, 4)
        item["score"] = round(new_score, 4)
        item["confidence_delta"] = delta
        item["explanation"] += explanation_suffix + f" Confidence delta: {delta}."
        item["reason"] = explain_delta(concept, delta)

    return own_results

def explain_delta(concept, delta):
    if delta > 0.05:
        return f"{concept} gained confidence after peer alignment (+{delta})"
    elif delta < -0.05:
        return f"{concept} dropped in confidence due to peer disagreement ({delta})"
    else:
        return f"{concept} maintained stable confidence after peer feedback."

# Round 1 Listener
def listen_and_publish_initial():
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNELS["subscribe_round1"])
    print(f"[RelationAgent] Subscribed to '{CHANNELS['subscribe_round1']}'")

    for msg in pubsub.listen():
        if msg['type'] != 'message':
            continue

        try:
            extraction = json.loads(msg['data'].decode())
            print("[RelationAgent] Received input for Round 1.")
            results = compute_initial_scores(extraction)
            r.set("relation_last_results", json.dumps(results))
            r.publish(CHANNELS["publish_round1"], json.dumps(results))
            print("[RelationAgent] Published Round 1 Scores.")

        except Exception as e:
            print(f"[RelationAgent] Error in Round 1: {e}")

# Round 2 Listener
def listen_and_publish_adjusted():
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNELS["subscribe_round2"])
    print(f"[RelationAgent] Subscribed to peer channel '{CHANNELS['subscribe_round2']}'")

    for msg in pubsub.listen():
        if msg['type'] != 'message':
            continue

        try:
            peer_results = json.loads(msg['data'].decode())
            own_json = r.get("relation_last_results")
            if not own_json:
                print("[RelationAgent] ⚠️ No Round 1 data cached.")
                continue

            own_results = json.loads(own_json)
            adjusted = adjust_scores_with_peer_feedback(peer_results, own_results)
            r.publish(CHANNELS["publish_round2"], json.dumps(adjusted))
            print("[RelationAgent] Published Adjusted Round 2 Scores.")

        except Exception as e:
            print(f"[RelationAgent] Error in Round 2: {e}")

if __name__ == "__main__":
    threading.Thread(target=listen_and_publish_initial).start()
    threading.Thread(target=listen_and_publish_adjusted).start()
