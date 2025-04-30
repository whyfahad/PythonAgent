import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

GOAL_RELS = {"MotivatedByGoal", "Desires", "UsedFor", "Causes", "CausesDesire", "HasSubevent"}


def run_reasoning_agent(extraction_result):
    concepts = extraction_result["concepts"]
    embeddings = np.array(extraction_result["concept_embeddings"])
    sentence_emb = np.array(extraction_result["sentence_embedding"])
    relations = extraction_result["conceptnet_relations"]
    llm_goals = extraction_result.get("inferred_goals", {})  # LLM-generated fallback goals

    results = []
    max_rel = max(len(r) for r in relations.values()) or 1

    for i, concept in enumerate(concepts):
        sim = cosine_similarity([sentence_emb], [embeddings[i]])[0][0]
        rels = relations.get(concept, [])

        # --- Check ConceptNet goal-related relations ---
        conceptnet_goals = [
            target for rel, target in rels if rel in GOAL_RELS
        ]

        # --- If ConceptNet has no goals, fallback to LLM ---
        if conceptnet_goals:
            inferred = conceptnet_goals
            goal_source = "ConceptNet"
        else:
            inferred = llm_goals.get(concept, [])
            goal_source = "LLM (fallback)"

        relation_strength = len(rels) / max_rel
        score = 0.3 * sim + 0.7 * relation_strength

        explanation = (
            f"Score is based on similarity ({round(sim, 2)}) and {len(rels)} ConceptNet relations. "
            f"Goal source: {goal_source}."
        )

        results.append({
            "agent": "RelationAgent",
            "concept": concept,
            "score": round(score, 4),
            "original_score": round(score, 4),
            "inferred_goals": inferred,
            "relation_count": len(rels),
            "explanation": explanation,
            "confidence_delta": 0.0
        })

    return results

def adjust_relation_scores_with_peer(peer_summary, own_results):
    peer_map = {item["concept"]: set(item["inferred_goals"]) for item in peer_summary}

    for item in own_results:
        concept = item["concept"]
        own_goals = set(item["inferred_goals"])
        peer_goals = peer_map.get(concept, set())

        original_score = item["original_score"]
        new_score = original_score
        explanation_addon = ""

        if own_goals and peer_goals and own_goals.intersection(peer_goals):
            new_score *= 1.1
            explanation_addon = " Boosted due to goal agreement with peer."
        elif not own_goals and not peer_goals:
            new_score *= 0.8
            explanation_addon = " Penalized due to lack of inferred goals by both agents."
        else:
            explanation_addon = " No consensus on inferred goals."

        new_score = round(new_score, 4)
        item["score"] = new_score
        item["confidence_delta"] = round(new_score - original_score, 4)
        item["explanation"] += explanation_addon + f" Confidence delta: {item['confidence_delta']}."

    return own_results


def generate_explanation(concept, original_score, new_score):
    delta = round(new_score - original_score, 4)
    if delta > 0.05:
        return f"{concept} gained confidence after peer alignment (+{delta})"
    elif delta < -0.05:
        return f"{concept} dropped in confidence due to peer disagreement ({delta})"
    else:
        return f"{concept} maintained stable confidence after peer feedback."
