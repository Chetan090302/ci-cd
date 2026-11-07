
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Issue Similarity API")

client = genai.Client(
    api_key="xxx"
)

class SimilarityRequest(BaseModel):
    sentences: list[str]

@app.post("/similarity")
def calculate_similarity(request: SimilarityRequest):
    sentences = request.sentences

    if len(sentences) < 2:
        raise HTTPException(
            status_code=400,
            detail="Provide at least two sentences"
        )

    if len(sentences) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 sentences per request"
        )

    if any(not s.strip() for s in sentences):
        raise HTTPException(
            status_code=400,
            detail="Sentences cannot be empty"
        )

    embeddings = []

    for sentence in sentences:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=sentence
        )

        embeddings.append(result.embeddings[0].values)

    vectors = np.asarray(embeddings)

    scores = cosine_similarity(vectors)

    return {
        "sentences": sentences,
        "similarity_matrix": scores.tolist()
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

def add(num1,num2):
    return num1+num2
