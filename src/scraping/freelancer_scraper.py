from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import time, random, re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from webdriver_manager.chrome import ChromeDriverManager


class ScrapingError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    base_url: str = "https://www.freelancer.com"
    delay_between_requests: tuple = (2, 5)
    max_pages_per_category: int = 3
    headless: bool = False
    wait_timeout: int = 15


@dataclass
class Project:
    platform: str
    category: str
    title: str
    description: Optional[str]
    skills: List[str]
    budget: Optional[str]
    time_left: Optional[str]
    bids_count: int
    url: str
    scraped_at: str


class FreelancerScraper:
    def __init__(self, config: Config):
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None

    # ✅ Context manager: with FreelancerScraper(...) as s:
    def __enter__(self) -> "FreelancerScraper":
        self.driver = self._setup_driver(self.config.headless)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.driver:
            self.driver.quit()
            self.driver = None

    def scrape_projects(self, categories: Dict[str, str], pages_per_category: Optional[int] = None) -> List[dict]:
        if not self.driver:
            raise ScrapingError("Driver not initialized. Use 'with FreelancerScraper(...) as s:'")

        pages = pages_per_category or self.config.max_pages_per_category

        all_projects: List[Project] = []
        for category_name, url_suffix in categories.items():
            category_url = f"{self.config.base_url}/{url_suffix}".replace("//jobs", "/jobs")
            all_projects.extend(self._scrape_category(category_name, category_url, pages))
            self._random_delay(3, 6)

        # de-dup by URL
        seen = set()
        unique: List[Project] = []
        for p in all_projects:
            if p.url not in seen:
                seen.add(p.url)
                unique.append(p)

        # Return list[dict] (keeps compatibility with your CSV pipeline)
        return [p.__dict__ for p in unique]

    # ----------------- internal helpers (encapsulation) -----------------

    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver

    def _random_delay(self, min_sec, max_sec):
        time.sleep(random.uniform(min_sec, max_sec))

    def _clean_text(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else None

    def _safe_find_text(self, parent, selector, default=None):
        try:
            el = parent.find_element(By.CSS_SELECTOR, selector)
            txt = self._clean_text(el.text)
            return txt if txt is not None else default
        except Exception:
            return default

    def _safe_find_elements(self, parent, selector):
        try:
            return parent.find_elements(By.CSS_SELECTOR, selector)
        except Exception:
            return []

    def _extract_project(self, card, category: str) -> Optional[Project]:
        # Title + URL
        try:
            title_elem = card.find_element(By.CSS_SELECTOR, "a.JobSearchCard-primary-heading-link")
        except Exception:
            try:
                title_elem = card.find_element(By.CSS_SELECTOR, "a[href*='/projects/']")
            except Exception:
                return None

        title = self._clean_text(title_elem.text if title_elem else None)
        url = title_elem.get_attribute("href") if title_elem else None
        if not title or not url:
            return None

        # Description
        desc = self._safe_find_text(card, "p.JobSearchCard-primary-description", default=None)
        if desc is None:
            desc = self._safe_find_text(card, ".ProjectSearchCard-secondary-heading", default=None)

        # Skills
        skill_elements = self._safe_find_elements(card, "div.JobSearchCard-primary-tags a.JobSearchCard-primary-tagsLink")
        if not skill_elements:
            skill_elements = self._safe_find_elements(card, "a[class*='tagsLink']")

        skills = []
        for el in skill_elements:
            t = self._clean_text(el.text)
            if t:
                skills.append(t)

        # Budget
        budget = None
        try:
            budget_el = card.find_element(By.CSS_SELECTOR, "div.JobSearchCard-primary-price")
            raw_budget = (budget_el.get_attribute("textContent") or "").strip()
            m = re.search(r"(\$|€|£)\s?\d[\d,]*(?:\s?-\s?\d[\d,]*)?", raw_budget)
            budget = m.group(0) if m else self._clean_text(raw_budget)
        except Exception:
            budget = None

        # Time left
        time_left = self._safe_find_text(card, "span.JobSearchCard-primary-heading-days", default=None)

        # Bids count
        bids_raw = self._safe_find_text(card, "div.JobSearchCard-secondary-entry", default=None)
        bids = 0
        if bids_raw:
            m = re.search(r"\d+", bids_raw)
            bids = int(m.group(0)) if m else 0

        return Project(
            platform="Freelancer.com",
            category=category,
            title=title,
            description=desc,
            skills=skills,
            budget=budget,
            time_left=time_left,
            bids_count=bids,
            url=url,
            scraped_at=datetime.now().isoformat(timespec="seconds"),
        )

    def _scrape_category(self, category_name: str, category_url: str, max_pages: int) -> List[Project]:
        assert self.driver is not None

        collected: List[Project] = []
        for page in range(1, max_pages + 1):
            page_url = category_url if page == 1 else f"{category_url}/{page}"
            self.driver.get(page_url)

            # Wait for cards
            try:
                WebDriverWait(self.driver, self.config.wait_timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.JobSearchCard-item"))
                )
                time.sleep(2)
            except TimeoutException:
                continue

            cards = self.driver.find_elements(By.CSS_SELECTOR, "div.JobSearchCard-item")
            if not cards:
                continue

            for card in cards:
                p = self._extract_project(card, category_name)
                if p:
                    collected.append(p)

            if page < max_pages:
                self._random_delay(*self.config.delay_between_requests)

        return collected
