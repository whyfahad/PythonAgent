import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def run_reasoning_agent(extraction_result):
    concepts = extraction_result["concepts"]
    embeddings = np.array(extraction_result["concept_embeddings"])
    sentence_emb = np.array(extraction_result["sentence_embedding"])
    relations = extraction_result["conceptnet_relations"]

    results = []
    max_rel = max(len(r) for r in relations.values()) or 1

    for i, concept in enumerate(concepts):
        sim = cosine_similarity([sentence_emb], [embeddings[i]])[0][0]
        rels = relations.get(concept, [])
        relation_strength = len(rels) / max_rel
        score = 0.3 * sim + 0.7 * relation_strength

        # Attempt to infer goals from ConceptNet relations
        inferred = [tgt for rel, tgt in rels if rel in [
            "MotivatedByGoal", "Desires", "HasSubevent", "UsedFor", "CausesDesire", "Causes"
        ]]
        goal_source = "ConceptNet" if inferred else "LLM-Fallback"

        # If ConceptNet fails, use a fallback label (not actual LLM logic here)
        if not inferred:
            inferred = []  # You could later fetch from LLM here

        explanation = f"Score is based on similarity ({round(sim, 2)}) and {len(rels)} ConceptNet relations."

        results.append({
            "agent": "RelationAgent",
            "concept": concept,
            "score": round(score, 4),
            "original_score": round(score, 4),
            "inferred_goals": inferred,
            "goal_source": goal_source,
            "relation_count": len(rels),
            "explanation": explanation,
            "confidence_delta": 0.0
        })

    return results
