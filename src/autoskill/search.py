"""Semantic search over skills and past messages using TF-IDF."""

from __future__ import annotations

import json
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import get_db


def _build_corpus(conn, include: Literal["all", "skills", "messages"] = "all") -> list[dict]:
    """Build a searchable corpus from the database."""
    corpus: list[dict] = []

    if include in ("all", "skills"):
        for row in conn.execute(
            "SELECT id, title, description, steps, tags, project_path, obs_type, facts, concepts FROM skills"
        ).fetchall():
            text = f"{row['title']} {row['description']} {row['steps']} {row['tags']} {row['facts'] or ''} {row['concepts'] or ''}"
            corpus.append({
                "type": "skill",
                "id": row["id"],
                "text": text,
                "title": row["title"],
                "description": row["description"],
                "project": row["project_path"],
                "obs_type": row["obs_type"] or "workflow",
            })

    if include in ("all", "messages"):
        for row in conn.execute(
            "SELECT m.id, m.user_text, m.response_summary, s.project_path "
            "FROM messages m JOIN sessions s ON m.session_id = s.id "
            "WHERE m.user_text IS NOT NULL"
        ).fetchall():
            text = f"{row['user_text']} {row['response_summary'] or ''}"
            corpus.append({
                "type": "message",
                "id": row["id"],
                "text": text,
                "title": row["user_text"][:100],
                "description": (row["response_summary"] or "")[:200],
                "project": row["project_path"],
            })

    return corpus


def search(
    query: str,
    db_path=None,
    include: Literal["all", "skills", "messages"] = "all",
    project: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search skills and messages by semantic similarity.

    Returns list of dicts with keys: type, id, title, description, project, score.
    """
    conn = get_db(db_path)
    corpus = _build_corpus(conn, include)
    conn.close()

    if not corpus:
        return []

    if project:
        corpus = [c for c in corpus if c.get("project") and project in c["project"]]
        if not corpus:
            return []

    texts = [c["text"] for c in corpus]
    texts.append(query)  # query is the last element

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)

    query_vec = tfidf[-1]
    corpus_vecs = tfidf[:-1]

    scores = cosine_similarity(query_vec, corpus_vecs).flatten()

    ranked = sorted(
        zip(range(len(corpus)), scores), key=lambda x: -x[1]
    )

    results = []
    for idx, score in ranked[:limit]:
        if score < 0.05:
            break
        item = corpus[idx]
        results.append({
            "type": item["type"],
            "id": item["id"],
            "title": item["title"],
            "description": item["description"],
            "project": item["project"],
            "score": round(float(score), 3),
        })

    return results
