import redis
import json
import time
import numpy as np
import spacy
import requests
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Redis client
r = redis.Redis(host='localhost', port=6379, db=0)
channel_subscribe = "user_input"
channel_publish = "concept_extracted"

# Load NLP and embedding models
nlp = spacy.load('en_core_web_sm')
embedding_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device("cpu"),
    dtype=torch.float32
)

# ---------------------- Utilities ---------------------- #
def extract_contextual_concepts(text):
    doc = nlp(text)
    concepts = set()
    for chunk in doc.noun_chunks:
        if len(chunk) > 1 or chunk.root.pos_ not in ['DET', 'PRON']:
            concepts.add(chunk.text.lower())
    concepts.update(ent.text.lower() for ent in doc.ents)
    concepts.update(
        token.lemma_.lower() for token in doc
        if token.pos_ in ['VERB', 'ADJ', 'ADV', 'PROPN'] and not token.is_stop and token.is_alpha
    )
    return list({c.strip().lower() for c in concepts if len(c) > 1})

def get_conceptnet_info(concept, limit=5):
    results = []
    words = concept.split()
    pos_filter = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
    filtered_words = [token.text for token in nlp(concept) if token.pos_ in pos_filter]
    query_words = filtered_words if filtered_words else words

    for word in query_words:
        url = f"http://api.conceptnet.io/c/en/{word.replace(' ', '_')}"
        response = requests.get(url)
        if response.status_code == 200:
            edges = response.json().get('edges', [])
            results.extend([
                (e['rel']['label'], e['end']['label'])
                for e in edges[:limit]
                if e['rel']['label'] not in ['ExternalURL', 'dbpedia']
            ])
        if len(results) >= limit:
            break
    return results[:limit]

def get_sentence_embedding(text):
    emb = embedding_model.predict([text], source_lang="eng_Latn")
    return emb[0]

def get_concept_embeddings(concepts):
    if not concepts:
        return np.array([])
    return embedding_model.predict(concepts, source_lang="eng_Latn")

def rank_concepts_by_similarity(sentence_embedding, concepts, concept_embeddings):
    if len(concepts) == 0 or len(concept_embeddings) == 0:
        return []
    similarities = cosine_similarity([sentence_embedding], concept_embeddings)[0]
    return sorted(zip(concepts, map(float, similarities)), key=lambda x: x[1], reverse=True)

def categorize_concepts(concepts):
    categories = {"needs": [], "objects": [], "actions": [], "descriptors": []}
    for c in concepts:
        for token in nlp(c):
            if token.pos_ == "NOUN":
                categories["objects"].append(c)
            elif token.pos_ == "VERB":
                categories["actions"].append(c)
            elif token.pos_ == "ADJ":
                categories["descriptors"].append(c)
            elif token.lemma_ in ["need", "want", "require", "wish"]:
                categories["needs"].append(c)
    return categories

# ---------------------- Main Agent Loop ---------------------- #
def run_concept_agent():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_subscribe)
    print(f"[ConceptAgent] Listening on channel: '{channel_subscribe}'...")

    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        try:
            text = message['data'].decode('utf-8')
            print(f"[ConceptAgent] Received input: {text}")

            # Process input
            concepts = extract_contextual_concepts(text)
            sentence_embedding = get_sentence_embedding(text)
            concept_embeddings = get_concept_embeddings(concepts)
            conceptnet_knowledge = {c: get_conceptnet_info(c) for c in concepts}
            ranked_concepts = rank_concepts_by_similarity(sentence_embedding, concepts, concept_embeddings)
            categorized_concepts = categorize_concepts(concepts)

            output = {
                "input": text,
                "concepts": concepts,
                "ranked_concepts": ranked_concepts,
                "conceptnet_relations": conceptnet_knowledge,
                "categorized_concepts": categorized_concepts,
                "sentence_embedding": sentence_embedding.tolist(),
                "concept_embeddings": [e.tolist() for e in concept_embeddings]
            }

            r.publish(channel_publish, json.dumps(output))
            print(f"[ConceptAgent] Published extracted concepts to '{channel_publish}'")

        except Exception as e:
            print(f"[ConceptAgent] Error: {str(e)}")

if __name__ == "__main__":
    run_concept_agent()
