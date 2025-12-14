from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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


class TfidfMatcher:
    """
    OOP wrapper around TF-IDF + cosine similarity.
    Keeps the vectorizer as object state (good POO signal).
    """

    def __init__(self, max_features: int = 5000, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
        self._X = None  # TF-IDF matrix for projects
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        """Fit TF-IDF vectorizer on the project corpus and store matrix."""
        self._X = self.vectorizer.fit_transform(corpus)
        self._fitted = True

    def rank(self, query_text: str) -> List[float]:
        """Return similarity scores between query and the fitted corpus."""
        if not self._fitted or self._X is None:
            raise ValueError("Matcher not fitted. Call fit(corpus) first.")

        q = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q, self._X).flatten()
        return sims.tolist()

    def match(
        self,
        projects_df: pd.DataFrame,
        profile_text: str,
        top_k: int = 10,
        text_col: str = "match_text",
    ) -> pd.DataFrame:
        """
        Fit on projects_df[text_col] then return top_k matches sorted by similarity.
        """
        if text_col not in projects_df.columns:
            raise ValueError(f"projects_df must contain '{text_col}'. Run cleaning first.")

        corpus = projects_df[text_col].fillna("").astype(str).tolist()
        self.fit(corpus)
        sims = self.rank(profile_text)

        out = projects_df.copy()
        out["score"] = sims

        cols = ["score", "title", "category", "budget", "time_left", "bids_count", "url"]
        for c in cols:
            if c not in out.columns:
                out[c] = "" if c != "bids_count" else 0

        out = out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        return out[cols]
