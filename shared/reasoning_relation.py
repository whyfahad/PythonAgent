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
        score = 0.3 * sim + 0.7 * (len(rels) / max_rel)
        inferred = [tgt for rel, tgt in rels if rel in ["MotivatedByGoal", "Desires"]]

        results.append({
            "agent": "RelationAgent",
            "concept": concept,
            "score": round(score, 4),
            "inferred_goals": inferred,
            "relation_count": len(rels),
            "inference": f"{concept} has {len(inferred)} goal(s) (Relation-weighted)"
        })
    return results
