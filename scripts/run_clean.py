import pandas as pd
from src.processing.clean_projects import clean_projects
from src.config import RAW_CSV_PATH, CLEAN_CSV_PATH


def main():
    df = pd.read_csv(RAW_CSV_PATH)
    df_clean = clean_projects(df)
    df_clean.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved cleaned data to: {CLEAN_CSV_PATH} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
