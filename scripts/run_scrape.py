import pandas as pd

from src.scraping.freelancer_scraper import Config, FreelancerScraper
from src.config import RAW_CSV_PATH


def main():
    categories = {
        "Python Development": "jobs/python/",
        "Web Development": "jobs/website-design/",
        "Data Entry": "jobs/data-entry/",
        "Graphic Design": "jobs/graphic-design/",
        "Mobile Apps": "jobs/mobile-phone/",
    }

    cfg = Config(
        headless=False,
        max_pages_per_category=3,
    )

    with FreelancerScraper(cfg) as scraper:
        rows = scraper.scrape_projects(categories)

    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved raw data to: {RAW_CSV_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
