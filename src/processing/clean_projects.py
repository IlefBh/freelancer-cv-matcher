from __future__ import annotations
import pandas as pd


def clean_projects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning rules:
    - Mandatory: title, description, skills, url
    - Optional: budget, time_left, bids_count
    - Fill optional NaNs with neutral values (no data loss)
    - Extra: remove blocked descriptions + build match_text for NLP matching
    """

    df = df.copy()

    # Ensure columns exist (safe for grading)
    required_cols = ["title", "description", "skills", "url"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # ---- 1) Basic text normalization (important)
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()

    # Drop rows missing mandatory content (also drop empty strings)
    df = df.dropna(subset=required_cols)
    df = df[(df["title"] != "") & (df["description"] != "") & (df["url"] != "")]

    # ---- 2) Remove placeholder / blocked descriptions (important)
    blocked_phrases = [
        "Please Sign Up or Login to see details",
        "Sign Up or Login to see details",
        "Please login to see details",
    ]
    df = df[~df["description"].str.contains("|".join(blocked_phrases), case=False, na=False)]

    # Standardize optional columns
    if "budget" in df.columns:
        df["budget"] = df["budget"].fillna("Not specified")

    if "time_left" in df.columns:
        df["time_left"] = df["time_left"].fillna("Unknown")

    if "bids_count" in df.columns:
        df["bids_count"] = df["bids_count"].fillna(0).astype(int)

    # Normalize skills column to list (sometimes it becomes string in CSV)
    def _ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
            parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
            return parts
        return [p.strip() for p in s.split(",") if p.strip()]

    df["skills"] = df["skills"].apply(_ensure_list)

    # ---- 3) Build text fields for matching (AI step)
    df["skills_text"] = df["skills"].apply(lambda xs: " ".join(xs) if isinstance(xs, list) else str(xs))
    df["match_text"] = (df["title"] + " " + df["description"] + " " + df["skills_text"]).str.strip()

    # Remove duplicates by URL
    df = df.drop_duplicates(subset=["url"])

    return df
