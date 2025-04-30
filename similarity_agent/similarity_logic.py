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
        rounded_score = round(score, 4)  # Compute once to avoid rounding inconsistencies

        results.append({
            "agent": "SimilarityAgent",
            "concept": concept,
            "score": rounded_score,              # Used for Round 1 scoring
            "original_score": rounded_score,     # Preserved baseline for Round 2 delta
            "confidence_delta": 0.0,             # Initially 0
            "inferred_goals": inferred,
            "relation_count": len(rels),
            "explanation": f"Score is based on similarity ({round(sim, 2)}) and {len(rels)} ConceptNet relations."
        })

    return results


def adjust_similarity_scores_with_peer(peer_summary, own_results):
    peer_map = {item["concept"]: set(item["inferred_goals"]) for item in peer_summary}

    for item in own_results:
        concept = item["concept"]
        own_goals = set(item.get("inferred_goals", []))
        peer_goals = peer_map.get(concept, set())
        original_score = item.get("original_score", item["score"])  # Safety fallback

        explanation_addon = ""
        updated_score = original_score

        if own_goals and peer_goals and own_goals.intersection(peer_goals):
            updated_score = round(original_score * 1.1, 4)
            explanation_addon = " Boosted due to goal agreement with peer."
        elif not own_goals and not peer_goals:
            updated_score = round(original_score * 0.8, 4)
            explanation_addon = " Penalized due to lack of inferred goals by both agents."
        else:
            explanation_addon = " No consensus on inferred goals."

        # Update final score and delta
        item["score"] = updated_score
        item["confidence_delta"] = round(updated_score - original_score, 4)
        item["reason"] = generate_explanation(concept, original_score, updated_score)
        item["explanation"] += explanation_addon + f" Confidence delta: {item['confidence_delta']}."

    return own_results

def generate_explanation(concept, original_score, new_score):
    delta = round(new_score - original_score, 4)
    if delta > 0.05:
        return f"{concept} became more relevant after peer influence (+{delta})"
    elif delta < -0.05:
        return f"{concept} lost confidence after peer disagreement ({delta})"
    else:
        return f"{concept} maintained stable confidence after peer feedback."
