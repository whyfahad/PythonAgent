# sonar_api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.sonar import load_model, encode_texts

app = FastAPI()
model = load_model("sonar_text", language="en")

class SentenceInput(BaseModel):
    sentences: List[str]

@app.post("/embed")
def get_embeddings(payload: SentenceInput):
    embeddings = encode_texts(model, payload.sentences)
    return {"embeddings": embeddings.tolist()}
