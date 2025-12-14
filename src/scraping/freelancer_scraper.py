from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import time
import random
import re
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


@dataclass
class Config:
    BASE_URL: str = "https://www.freelancer.com"
    DELAY_BETWEEN_REQUESTS: tuple = (2, 5)
    MAX_PAGES_PER_CATEGORY: int = 3
    HEADLESS: bool = False


def setup_driver(headless: bool = False) -> webdriver.Chrome:
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


def _clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def safe_find_element_text(parent, selector, by=By.CSS_SELECTOR, default=None):
    try:
        element = parent.find_element(by, selector)
        txt = _clean_text(element.text)
        return txt if txt is not None else default
    except Exception:
        return default


def safe_find_elements(parent, selector, by=By.CSS_SELECTOR):
    try:
        return parent.find_elements(by, selector)
    except Exception:
        return []


def random_delay(min_sec, max_sec):
    time.sleep(random.uniform(min_sec, max_sec))


def extract_project_data(project_card, category: str) -> Optional[Dict]:
    project = {
        "platform": "Freelancer.com",
        "category": category,
        "title": None,
        "description": None,
        "skills": [],
        "budget": None,
        "time_left": None,
        "bids_count": None,
        "url": None,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }

    # --- Title + URL (primary)
    title_elem = None
    try:
        title_elem = project_card.find_element(By.CSS_SELECTOR, "a.JobSearchCard-primary-heading-link")
    except Exception:
        # fallback selector
        try:
            title_elem = project_card.find_element(By.CSS_SELECTOR, "a[href*='/projects/']")
        except Exception:
            return None

    title = _clean_text(title_elem.text if title_elem else None)
    href = title_elem.get_attribute("href") if title_elem else None

    if not title or not href:
        return None

    project["title"] = title
    project["url"] = href

    # --- Description (JobSearchCard layout)
    project["description"] = safe_find_element_text(project_card, "p.JobSearchCard-primary-description", default=None)

    # fallback description selector (if layout changed)
    if project["description"] is None:
        project["description"] = safe_find_element_text(project_card, ".ProjectSearchCard-secondary-heading", default=None)

    # --- Skills / tags
    skill_elements = safe_find_elements(project_card, "div.JobSearchCard-primary-tags a.JobSearchCard-primary-tagsLink")
    if not skill_elements:
        skill_elements = safe_find_elements(project_card, "a[class*='tagsLink']")

    skills = []
    for el in skill_elements:
        t = _clean_text(el.text)
        if t:
            skills.append(t)
    project["skills"] = skills


    # --- Budget (robust + cleaned)
    project["budget"] = None
    try:
        budget_el = project_card.find_element(By.CSS_SELECTOR, "div.JobSearchCard-primary-price")
        raw_budget = (budget_el.get_attribute("textContent") or "").strip()

        # Keep only the money part (avoids "\nAverage bid")
        m = re.search(r"(\$|€|£)\s?\d[\d,]*(?:\s?-\s?\d[\d,]*)?", raw_budget)
        project["budget"] = m.group(0) if m else _clean_text(raw_budget)

    except Exception:
        project["budget"] = None


   # --- Time left (deadline)
    project["time_left"] = safe_find_element_text(
        project_card,
        "span.JobSearchCard-primary-heading-days",
        default=None,
    )


   # --- Bids count (competition level)
    bids_raw = safe_find_element_text(
        project_card,
        "div.JobSearchCard-secondary-entry",
        default=None,
    )

    if bids_raw:
        m = re.search(r"\d+", bids_raw)
        project["bids_count"] = int(m.group(0)) if m else None
    else:
        project["bids_count"] = None

    return project



def scrape_category(driver, category_name: str, category_url: str, max_pages: int) -> List[Dict]:
    all_projects: List[Dict] = []

    for page in range(1, max_pages + 1):
        page_url = category_url if page == 1 else f"{category_url}/{page}"

        driver.get(page_url)

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.JobSearchCard-item"))
            )
            time.sleep(2)
        except TimeoutException:
            # If JobSearchCard layout not found, try fallback layout
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/projects/']"))
                )
            except TimeoutException:
                continue

        cards = driver.find_elements(By.CSS_SELECTOR, "div.JobSearchCard-item")
        if not cards:
            # fallback: attempt to use a broader container
            cards = driver.find_elements(By.CSS_SELECTOR, "div:has(a[href*='/projects/'])")

        for card in cards:
            p = extract_project_data(card, category_name)
            if p:
                all_projects.append(p)

        if page < max_pages:
            random_delay(*Config.DELAY_BETWEEN_REQUESTS)

    return all_projects


def scrape_projects(categories: Dict[str, str], pages_per_category: int = 3, headless: bool = False) -> List[Dict]:
    driver = setup_driver(headless=headless)
    all_projects: List[Dict] = []

    try:
        for category_name, url_suffix in categories.items():
            category_url = f"{Config.BASE_URL}/{url_suffix}".replace("//jobs", "/jobs")
            all_projects.extend(scrape_category(driver, category_name, category_url, pages_per_category))
            random_delay(3, 6)
    finally:
        driver.quit()

    # de-dup by URL
    seen = set()
    unique = []
    for p in all_projects:
        u = p.get("url")
        if u and u not in seen:
            seen.add(u)
            unique.append(p)

    return unique
