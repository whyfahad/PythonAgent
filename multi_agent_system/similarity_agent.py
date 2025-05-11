import redis
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe_r1 = "concept_extracted"
channel_publish_r1 = "similarity_scored"
channel_subscribe_r2 = "peer_similarity"
channel_publish_r2 = "similarity_adjusted"
cache_key = "similarity_last_results"

# ---------------------- Round 1 Scoring ---------------------- #
def run_similarity_agent(data):
    concepts = data["concepts"]
    embeddings = np.array(data["concept_embeddings"])
    sentence_emb = np.array(data["sentence_embedding"])
    relations = data["conceptnet_relations"]
    llm_goals = data.get("inferred_goals", {})

    results = []
    max_rel = max((len(v) for v in relations.values()), default=1)

    for i, concept in enumerate(concepts):
        sim = cosine_similarity([sentence_emb], [embeddings[i]])[0][0]
        rels = relations.get(concept, [])
        inferred_goals = llm_goals.get(concept, [])

        score = 0.8 * sim + 0.2 * (len(rels) / max_rel)
        rounded_score = round(score, 4)

        results.append({
            "agent": "SimilarityAgent",
            "concept": concept,
            "score": rounded_score,
            "original_score": rounded_score,
            "confidence_delta": 0.0,
            "inferred_goals": inferred_goals,
            "relation_count": len(rels),
            "explanation": f"Score based on similarity ({round(sim, 2)}) and {len(rels)} ConceptNet relations.",
            "reason": f"Initial score from semantic similarity and relation count."
        })

    return results

# ---------------------- Round 2 Adjustment ---------------------- #
def adjust_scores_with_peer(peer_data, own_results):
    peer_map = {item["concept"]: set(item.get("inferred_goals", [])) for item in peer_data}

    for item in own_results:
        concept = item["concept"]
        own_goals = set(item.get("inferred_goals", []))
        peer_goals = peer_map.get(concept, set())
        original_score = item.get("original_score", item["score"])
        new_score = original_score
        delta_reason = ""

        if own_goals and peer_goals and own_goals & peer_goals:
            new_score *= 1.1
            delta_reason = " Boosted due to shared inferred goals with peer."
        elif not own_goals and not peer_goals:
            new_score *= 0.8
            delta_reason = " Penalized due to missing goals in both agents."
        else:
            delta_reason = " No consensus on inferred goals."

        new_score = round(new_score, 4)
        delta = round(new_score - original_score, 4)

        item["score"] = new_score
        item["confidence_delta"] = delta
        item["explanation"] += delta_reason + f" Confidence delta: {delta}."
        item["reason"] = generate_explanation(concept, original_score, new_score)

    return own_results

def generate_explanation(concept, original_score, new_score):
    delta = round(new_score - original_score, 4)
    if delta > 0.05:
        return f"{concept} gained confidence after peer alignment (+{delta})"
    elif delta < -0.05:
        return f"{concept} dropped in confidence due to peer disagreement ({delta})"
    else:
        return f"{concept} maintained stable confidence after peer feedback."

# ---------------------- Redis Listeners ---------------------- #
def round1_listener():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe_r1)
    print(f"[SimilarityAgent] 📡 Subscribed to '{channel_subscribe_r1}'")

    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        try:
            data = json.loads(msg["data"].decode())
            print("[SimilarityAgent]  Received Round 1 Input")
            results = run_similarity_agent(data)
            r.set(cache_key, json.dumps(results))
            r.publish(channel_publish_r1, json.dumps(results))
            print("[SimilarityAgent]  Published Round 1 Scores")
        except Exception as e:
            print("[SimilarityAgent]  Error in Round 1:", str(e))

def round2_listener():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe_r2)
    print(f"[SimilarityAgent] 📡 Subscribed to peer feedback: '{channel_subscribe_r2}'")

    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        try:
            peer_data = json.loads(msg["data"].decode())
            cached = r.get(cache_key)
            if not cached:
                print("[SimilarityAgent] ⚠️ No cached Round 1 data.")
                continue

            own_results = json.loads(cached)
            adjusted = adjust_scores_with_peer(peer_data, own_results)
            r.publish(channel_publish_r2, json.dumps(adjusted))
            print("[SimilarityAgent]  Published Adjusted Round 2 Scores")
        except Exception as e:
            print("[SimilarityAgent]  Error in Round 2:", str(e))

# ---------------------- Entry Point ---------------------- #
if __name__ == "__main__":
    threading.Thread(target=round1_listener).start()
    threading.Thread(target=round2_listener).start()
