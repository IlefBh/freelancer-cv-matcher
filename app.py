from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.cv_parser.cv_text import extract_cv_text, build_profile_text
from src.matching.tfidf_matcher import TfidfMatcher
from io import BytesIO
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 4),
    "figure.dpi": 110,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})



APP_TITLE = "Freelancer CV Matcher"
INPUTS_DIR = Path("inputs")
DEFAULT_CV_NAME = "my_cv"


def ensure_dirs():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_cv(uploaded_file) -> Path:
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".pdf", ".docx"]:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX.")

    out_path = INPUTS_DIR / f"{DEFAULT_CV_NAME}{suffix}"
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def load_projects() -> pd.DataFrame:
    p = Path("data/processed/freelancer_projects_clean.csv")
    if not p.exists():
        raise FileNotFoundError(
            "Missing data/processed/freelancer_projects_clean.csv. Run:\n"
            "  py -m scripts.run_scrape\n"
            "  py -m scripts.run_clean"
        )
    return pd.read_csv(p)


def ui_inject_css():
    st.markdown(
        """
        <style>
        /* Layout */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 14px;
            padding: 0.65rem 1.1rem;
            font-weight: 600;
            background: linear-gradient(135deg, #6C63FF, #4FD1C5);
            color: white;
            border: none;
        }

        /* Cards */
        .card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid #E5E7EB;
            background: #FFFFFF;
            margin-bottom: 0.9rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
            transition: all 0.2s ease-in-out;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(108,99,255,0.25);
        }

        /* Title */
        .title {
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }

        .title a {
            color: #111827;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 0.4rem;
            font-weight: 600;
            color: white;
        }

        .badge:nth-child(1) { background: #6C63FF; }
        .badge:nth-child(2) { background: #10B981; }
        .badge:nth-child(3) { background: #F59E0B; color: #1F2937; }
        .badge:nth-child(4) { background: #EF4444; }
        .badge:nth-child(5) { background: #3B82F6; }

        /* Score */
        .score {
            font-variant-numeric: tabular-nums;
            font-weight: 800;
            background: linear-gradient(90deg, #10B981, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Muted */
        .muted {
            color: #6B7280;
            font-size: 0.9rem;
        }

        a {
            text-decoration: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def plot_score_distribution(results):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        results["score"],
        bins=15,
        color="#6C63FF",
        edgecolor="white",
        alpha=0.85
    )

    ax.set_title("Distribution of similarity scores")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of projects")

    st.pyplot(fig)



def plot_topk_scores(results):
    top = results.head(10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top["title"],
        top["score"],
        color="#4FD1C5"
    )

    ax.invert_yaxis()
    ax.set_title("Top 10 matched projects")
    ax.set_xlabel("Cosine similarity")

    st.pyplot(fig)



def plot_score_vs_bids(results):
    if "bids_count" not in results.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(
        results["bids_count"],
        results["score"],
        color="#F59E0B",
        alpha=0.75,
        edgecolors="white"
    )

    ax.set_title("Match quality vs competition")
    ax.set_xlabel("Number of bids")
    ax.set_ylabel("Cosine similarity")

    st.pyplot(fig)



def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="centered")
    ui_inject_css()

    st.title("Freelancer CV Matcher")
    st.caption("Upload an English CV (PDF/DOCX) → we recommend the most relevant Freelancer projects using TF-IDF + cosine similarity.")

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top-K recommendations", 5, 30, 10, 1)
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.05, 0.01)
        max_bids = st.number_input("Max bids (0 = no filter)", min_value=0, value=0, step=5)
        category_filter = st.text_input("Category contains (optional)", value="")
        st.divider()
        st.write("Data source: `data/processed/freelancer_projects_clean.csv`")

    uploaded = st.file_uploader("Upload your CV", type=["pdf", "docx"])

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("🔎 Match projects", type="primary", use_container_width=True)
    with colB:
        st.button("🧹 Clear", use_container_width=True, on_click=lambda: st.session_state.clear())

    if uploaded is None:
        st.info("Upload a CV to start.")
        return

    try:
        saved_path = save_uploaded_cv(uploaded)
        st.success(f"Saved CV to: `{saved_path.as_posix()}`")
    except Exception as e:
        st.error(str(e))
        return

    if not run_btn:
        st.stop()

    # Load data + run matching
    try:
        projects = load_projects()
        cv_text = extract_cv_text(str(saved_path))
        profile_text = build_profile_text(cv_text)
        matcher = TfidfMatcher()
        results = matcher.match(projects, profile_text, top_k=top_k)
    except Exception as e:
        st.error(f"Error while matching: {e}")
        return

    # Optional filters on results
    if max_bids and "bids_count" in results.columns:
        results = results[results["bids_count"].fillna(0).astype(int) <= int(max_bids)]
    if category_filter.strip():
        results = results[results["category"].astype(str).str.contains(category_filter.strip(), case=False, na=False)]
    results = results[results["score"] >= float(min_score)].reset_index(drop=True)

    # Header metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Projects matched", f"{len(results)}")
    m2.metric("CV file", saved_path.name)
    m3.metric("Top score", f"{results['score'].max():.3f}" if len(results) else "—")

    st.divider()

    st.subheader("Match insights")

    plot_score_distribution(results)
    plot_topk_scores(results)
    plot_score_vs_bids(results)

    st.divider()


    if len(results) == 0:
        st.warning("No results match your filters. Try lowering the minimum score or removing filters.")
        return

   # Create Excel file in memory
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Matches")

    st.download_button(
        "⬇️ Download results (Excel)",
        data=excel_buffer.getvalue(),
        file_name="cv_project_matches.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Recommended projects")

    # Pretty cards
    for i, row in results.iterrows():
        title = str(row.get("title", "Untitled"))
        url = str(row.get("url", ""))
        score = float(row.get("score", 0.0))
        category = str(row.get("category", ""))
        budget = str(row.get("budget", "Not specified"))
        time_left = str(row.get("time_left", "Unknown"))
        bids = row.get("bids_count", 0)
        try:
            bids = int(bids)
        except Exception:
            bids = 0

        st.markdown(
            f"""
            <div class="card">
              <div class="title">
                <a href="{url}" target="_blank">{i+1}. {title}</a>
              </div>
              <div class="muted" style="margin-bottom:0.55rem;">
                <span class="badge">📌 {category}</span>
                <span class="badge">💰 {budget}</span>
                <span class="badge">⏳ {time_left}</span>
                <span class="badge">👥 {bids} bids</span>
                <span class="badge"><span class="score">🧠 {score:.3f}</span> score</span>
              </div>
              <div class="muted">{url}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Table view")
    st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
