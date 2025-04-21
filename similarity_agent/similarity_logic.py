import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def run_similarity_agent(extraction_result):
    concepts = extraction_result["concepts"]
    embeddings = np.array(extraction_result["concept_embeddings"])
    sentence_emb = np.array(extraction_result["sentence_embedding"])
    relations = extraction_result["conceptnet_relations"]
    extracted_goals = extraction_result.get("inferred_goals", {})

    results = []
    max_rel = max(len(r) for r in relations.values()) or 1

    for i, concept in enumerate(concepts):
        sim = cosine_similarity([sentence_emb], [embeddings[i]])[0][0]
        rels = relations.get(concept, [])
        inferred = extracted_goals.get(concept, [])

        score = 0.8 * sim + 0.2 * (len(rels) / max_rel)

        results.append({
            "agent": "SimilarityAgent",
            "concept": concept,
            "score": round(score, 4),
            "inferred_goals": inferred,
            "relation_count": len(rels),
            "inference": f"{concept} has {len(inferred)} goal(s) (Similarity-weighted)"
        })

    return results


def adjust_similarity_scores_with_peer(peer_summary, own_results):
    peer_map = {item["concept"]: set(item["inferred_goals"]) for item in peer_summary}

    for item in own_results:
        concept = item["concept"]
        own_goals = set(item["inferred_goals"])
        peer_goals = peer_map.get(concept, set())

        if own_goals and peer_goals:
            item["score"] = round(item["score"] * 1.1, 4)  # Boost if both agents found goals
        elif not own_goals and not peer_goals:
            item["score"] = round(item["score"] * 0.8, 4)  # Penalize if neither found goals

    return own_results
