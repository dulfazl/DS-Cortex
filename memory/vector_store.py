"""A small local vector store over the research notes. TF-IDF + cosine similarity keeps it
offline and dependency-free; swap in a local embedding model later if needed."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

NOTES_PATH = Path(__file__).resolve().parent / "research_notes.json"


@lru_cache(maxsize=1)
def _index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    data = json.loads(NOTES_PATH.read_text())
    notes = data["notes"]
    docs = [f"{n['title']} {n['summary']} {n['tags']} {n['tags']}" for n in notes]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs)
    return notes, vec, mat, data["note"]


def retrieve(query: str, k: int = 4) -> dict:
    notes, vec, mat, disclaimer = _index()
    q = vec.transform([query])
    sims = (mat @ q.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    hits = [{**notes[i], "similarity": round(float(sims[i]), 4)} for i in order if sims[i] > 0]
    return {"query": query, "hits": hits, "ids": [h["id"] for h in hits], "disclaimer": disclaimer}


def by_practice(practice: str) -> list[dict]:
    notes, *_ = _index()
    return [n for n in notes if n["practice"] == practice]
