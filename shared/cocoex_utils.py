import spacy
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy and sentence embedding model (all-mpnet-base-v2)
nlp = spacy.load('en_core_web_sm')
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# ---------------------- Concept Extraction ---------------------- #
def extract_contextual_concepts(text):
    doc = nlp(text)
    concepts = set()

    for chunk in doc.noun_chunks:
        if len(chunk) > 1 or chunk.root.pos_ not in ['DET', 'PRON']:
            concepts.add(chunk.text.lower())

    concepts.update(ent.text.lower() for ent in doc.ents)

    concepts.update(
        token.lemma_.lower() for token in doc
        if token.pos_ in ['VERB', 'ADJ', 'ADV', 'PROPN']
        and not token.is_stop
        and token.is_alpha
    )

    return list({c.strip().lower() for c in concepts if len(c) > 1})

# ---------------------- ConceptNet Lookup ---------------------- #
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

# ---------------------- Embedding using all-mpnet-base-v2 ---------------------- #
def get_sentence_embedding(text):
    return embedding_model.encode(text)

def get_concept_embeddings(concepts):
    return embedding_model.encode(concepts)

# ---------------------- Ranking ---------------------- #
def rank_concepts_by_similarity(sentence_embedding, concepts, concept_embeddings):
    if len(concepts) == 0 or len(concept_embeddings) == 0:
        return []
    similarities = cosine_similarity([sentence_embedding], concept_embeddings)[0]
    return sorted(zip(concepts, map(float, similarities)), key=lambda x: x[1], reverse=True)

# ---------------------- Categorization ---------------------- #
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

# ---------------------- Entry Point ---------------------- #
def run_extraction_agent(sentence):
    concepts = extract_contextual_concepts(sentence)
    sentence_embedding = get_sentence_embedding(sentence)
    concept_embeddings = get_concept_embeddings(concepts)
    conceptnet_knowledge = {concept: get_conceptnet_info(concept) for concept in concepts}
    ranked_concepts = rank_concepts_by_similarity(sentence_embedding, concepts, concept_embeddings)
    concept_categories = categorize_concepts(concepts)

    return {
        "concepts": concepts,
        "ranked_concepts": ranked_concepts,
        "conceptnet_relations": conceptnet_knowledge,
        "sentence_embedding": sentence_embedding.tolist(),
        "concept_embeddings": [e.tolist() for e in concept_embeddings],
        "categorized_concepts": concept_categories
    }@app.websocket("/extract")
async def extract(websocket: WebSocket):
    ...
    result = run_extraction_agent(text)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        else:
            return obj

    await websocket.send_json(convert(result))