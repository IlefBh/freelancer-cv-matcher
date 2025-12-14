import argparse
import pandas as pd

from src.cv_parser.cv_text import extract_cv_text, build_profile_text
from src.matching.tfidf_matcher import match_projects_tfidf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", required=True, help="Path to CV file (.pdf or .docx)")
    parser.add_argument("--k", type=int, default=10, help="Top K matches")
    args = parser.parse_args()

    projects = pd.read_csv("data/processed/freelancer_projects_clean.csv")
    cv_text = extract_cv_text(args.cv)
    profile_text = build_profile_text(cv_text)

    results = match_projects_tfidf(projects, profile_text, top_k=args.k)
    print(results.to_string(index=False))

    results.to_csv("data/processed/cv_project_matches.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: data/processed/cv_project_matches.csv")


if __name__ == "__main__":
    main()
