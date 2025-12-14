from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchResult:
    url: str
    title: str
    score: float
    category: str
    budget: str
    time_left: str
    bids_count: int


def match_projects_tfidf(
    projects_df: pd.DataFrame,
    profile_text: str,
    top_k: int = 10,
) -> pd.DataFrame:
    if "match_text" not in projects_df.columns:
        raise ValueError("projects_df must contain a 'match_text' column. Run cleaning first.")

    corpus = projects_df["match_text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    q = vectorizer.transform([profile_text])

    sims = cosine_similarity(q, X).flatten()
    out = projects_df.copy()
    out["score"] = sims

    cols = ["score", "title", "category", "budget", "time_left", "bids_count", "url"]
    for c in cols:
        if c not in out.columns:
            out[c] = "" if c != "bids_count" else 0

    out = out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return out[cols]
