"""
arXiv paper fetcher — smart incremental mode
  首次运行（无历史数据）: 全量抓取 START_YEAR ~ 今年，按年并行
  后续运行（有历史数据）: 只抓最近 FETCH_RECENT_DAYS 天，速度快、降低限速风险
  --full : 强制全量重抓（用于恢复/修复）
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from config import SEARCH_QUERIES, START_YEAR, MAX_RESULTS, FETCH_RECENT_DAYS
from config_search import ArxivScraper

END_YEAR = datetime.now().year
OUTPUT_DIR = "output"
PARALLEL_WORKERS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_existing_papers() -> dict:
    """Load cached papers from progress.json → {id: paper}"""
    path = os.path.join(OUTPUT_DIR, "progress.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        papers = {p["id"]: p for p in data.get("papers", [])}
        logger.info(f"Loaded {len(papers)} existing papers from cache")
        return papers
    except Exception as e:
        logger.warning(f"Could not load progress.json: {e}")
        return {}


def save_papers(papers: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "progress.json"), "w") as f:
        json.dump({"last_update": datetime.now().isoformat(), "papers": papers}, f, indent=2)
    logger.info(f"Saved {len(papers)} papers to cache")


# ── fetch strategies ──────────────────────────────────────────────────────────

def _fetch_one(args_tuple):
    """Worker: fetch papers for one (query, date_filter) pair."""
    query, date_filter, max_results, delay = args_tuple
    scraper = ArxivScraper(delay=delay)
    full_query = f"({query}) AND {date_filter}"
    papers = scraper.fetch_all(full_query, max_total=max_results)
    return query[:40], papers


def fetch_recent(existing: dict, days: int = FETCH_RECENT_DAYS) -> dict:
    """Incremental: fetch only the last `days` days across all queries."""
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    today  = date.today().strftime("%Y%m%d")
    date_filter = f"submittedDate:[{cutoff} TO {today}]"

    logger.info(f"Incremental fetch: last {days} days ({cutoff} → {today})")
    logger.info(f"Queries: {len(SEARCH_QUERIES)}")

    all_papers = existing.copy()
    tasks = [(q, date_filter, 2000, 3.0) for q in SEARCH_QUERIES]

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            label, papers = future.result()
            new = sum(1 for p in papers if p["id"] not in all_papers)
            for p in papers:
                all_papers[p["id"]] = p          # always overwrite with latest
            logger.info(f"  '{label}…': {len(papers)} fetched, {new} new")

    return all_papers


def fetch_full(existing: dict) -> dict:
    """Full fetch: all years from START_YEAR to now, one year at a time."""
    # Detect which years are already complete
    existing_years: set = set()
    for p in existing.values():
        try:
            existing_years.add(int(p["published"][:4]))
        except Exception:
            pass

    all_years = list(range(START_YEAR, END_YEAR + 1))
    years_to_fetch = [y for y in all_years if y not in existing_years]
    # Always re-fetch current year (may have new papers)
    if END_YEAR not in years_to_fetch:
        years_to_fetch.append(END_YEAR)

    logger.info(f"Full fetch: years {years_to_fetch}")
    all_papers = existing.copy()

    for year in years_to_fetch:
        logger.info(f"=== Year {year} ===")
        date_filter = f"submittedDate:[{year}0101 TO {year}1231]"
        tasks = [(q, date_filter, MAX_RESULTS, 3.0) for q in SEARCH_QUERIES]
        year_new = 0

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                label, papers = future.result()
                new = 0
                for p in papers:
                    if p["id"] not in all_papers:
                        all_papers[p["id"]] = p
                        new += 1
                year_new += new
                logger.info(f"  '{label}…': {len(papers)} fetched, {new} new")

        logger.info(f"Year {year} done: {year_new} new papers")

    return all_papers


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv")
    parser.add_argument("--full", action="store_true",
                        help="Force full refetch (all years). Default: incremental if cache exists.")
    args = parser.parse_args()

    existing = load_existing_papers()

    if existing and not args.full:
        # Have history → incremental (last 30 days)
        all_papers = fetch_recent(existing)
    else:
        # First run or --full → fetch all years
        all_papers = fetch_full(existing)

    papers_list = list(all_papers.values())
    save_papers(papers_list)
    logger.info(f"Done. Total papers in cache: {len(papers_list)}")
    return papers_list


if __name__ == "__main__":
    main()
