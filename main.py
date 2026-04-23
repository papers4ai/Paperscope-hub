"""
论文抓取工具 - 支持多数据源

数据源：
  --source arxiv     : 使用 arXiv API（官方，限制严格）
  --source openalex  : 使用 OpenAlex API（推荐，限制宽松）

模式：
  默认               : 增量更新（最近 30 天）
  --full             : 全量抓取（所有年份）
  --year 2025        : 只抓取指定年份
  --years 2023,2024  : 抓取多个年份

示例：
  python main.py                              # OpenAlex 增量更新
  python main.py --source arxiv               # arXiv 增量更新
  python main.py --source openalex --full     # OpenAlex 全量
  python main.py --year 2025                  # 只抓 2025 年
  python main.py --source arxiv --years 2023,2024  # arXiv 抓 2023-2024
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
from openalex_scraper import OpenAlexScraper
from keyword_generator import get_openalex_keywords

END_YEAR = datetime.now().year
OUTPUT_DIR = "output"

# OpenAlex 配置
OPENALEX_DELAY = 0.5

# arXiv 配置
ARXIV_DELAY = 15.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def load_existing_papers() -> dict:
    """Load cached papers from progress.json"""
    path = os.path.join(OUTPUT_DIR, "progress.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        papers = {p["id"]: p for p in data.get("papers", [])}
        logger.info(f"Loaded {len(papers)} existing papers")
        return papers
    except Exception as e:
        logger.warning(f"Could not load progress.json: {e}")
        return {}


def save_papers(papers: dict):
    """Save papers dict to progress.json"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    papers_list = list(papers.values())
    papers_list.sort(key=lambda x: x.get("published", ""), reverse=True)

    with open(os.path.join(OUTPUT_DIR, "progress.json"), "w") as f:
        json.dump({
            "last_update": datetime.now().isoformat(),
            "papers": papers_list
        }, f, indent=2)
    logger.info(f"Saved {len(papers_list)} papers to progress.json")


def get_existing_years(papers: dict) -> set:
    """Get years that already have data"""
    years = set()
    for p in papers.values():
        try:
            years.add(int(p["published"][:4]))
        except:
            pass
    return years


# ── OpenAlex Fetcher ───────────────────────────────────────────────────────

def fetch_openalex(papers: dict, years: list = None, incremental: bool = False) -> dict:
    """使用 OpenAlex 抓取"""
    scraper = OpenAlexScraper(delay=OPENALEX_DELAY)

    # 动态获取关键词
    keywords = get_openalex_keywords()
    logger.info(f"Using {len(keywords)} keywords from keyword_generator")

    all_papers = papers.copy()
    new_count = 0

    if incremental:
        # 增量模式：最近 30 天
        cutoff = (date.today() - timedelta(days=FETCH_RECENT_DAYS)).strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")
        logger.info(f"OpenAlex incremental: {cutoff} → {today}")

        for kw in keywords:
            try:
                results = scraper.search_by_keyword(kw, cutoff, today, max_results=200)
                for p in results:
                    if p["id"] not in all_papers:
                        all_papers[p["id"]] = p
                        new_count += 1
                logger.info(f"  '{kw}': {len(results)} papers")
            except Exception as e:
                logger.warning(f"  '{kw}': failed - {e}")

    else:
        # 按年份抓取
        for year in years:
            logger.info(f"=== OpenAlex Year {year} ===")
            year_new = 0
            from_date = f"{year}-01-01"
            to_date = f"{year}-12-31"

            for kw in keywords:
                try:
                    results = scraper.search_by_keyword(kw, from_date, to_date, max_results=500)
                    for p in results:
                        if p["id"] not in all_papers:
                            all_papers[p["id"]] = p
                            year_new += 1
                            new_count += 1
                    logger.info(f"  '{kw}': {len(results)} papers, {year_new} new this year")
                except Exception as e:
                    logger.warning(f"  '{kw}': failed - {e}")

            logger.info(f"Year {year} done: {year_new} new papers")

    logger.info(f"OpenAlex total: {new_count} new papers")
    return all_papers


# ── arXiv Fetcher ──────────────────────────────────────────────────────────

def fetch_arxiv(papers: dict, years: list = None, incremental: bool = False) -> dict:
    """使用 arXiv API 抓取"""
    all_papers = papers.copy()
    new_count = 0

    if incremental:
        # 增量模式
        cutoff = (date.today() - timedelta(days=FETCH_RECENT_DAYS)).strftime("%Y%m%d")
        today = date.today().strftime("%Y%m%d")
        date_filter = f"submittedDate:[{cutoff} TO {today}]"
        logger.info(f"arXiv incremental: {cutoff} → {today}")

        for query in SEARCH_QUERIES:
            try:
                scraper = ArxivScraper(delay=ARXIV_DELAY)
                full_query = f"({query}) AND {date_filter}"
                results = scraper.fetch_all(full_query, max_total=2000)

                for p in results:
                    if p["id"] not in all_papers:
                        all_papers[p["id"]] = p
                        new_count += 1

                logger.info(f"  Query: {len(results)} papers")
            except Exception as e:
                logger.warning(f"  Query failed: {e}")

    else:
        # 按年份抓取
        for year in years:
            logger.info(f"=== arXiv Year {year} ===")
            year_new = 0
            date_filter = f"submittedDate:[{year}0101 TO {year}1231]"

            for query in SEARCH_QUERIES:
                try:
                    scraper = ArxivScraper(delay=ARXIV_DELAY)
                    full_query = f"({query}) AND {date_filter}"
                    results = scraper.fetch_all(full_query, max_total=MAX_RESULTS)

                    for p in results:
                        if p["id"] not in all_papers:
                            all_papers[p["id"]] = p
                            year_new += 1
                            new_count += 1

                    logger.info(f"  Query: {len(results)} fetched, {year_new} new")
                except Exception as e:
                    logger.warning(f"  Query failed: {e}")

            logger.info(f"Year {year} done: {year_new} new papers")

    logger.info(f"arXiv total: {new_count} new papers")
    return all_papers


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="论文抓取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # OpenAlex 增量更新
  python main.py --update-keywords            # 更新关键词 + OpenAlex 增量更新
  python main.py --source arxiv               # arXiv 增量更新
  python main.py --full                       # OpenAlex 全量抓取
  python main.py --year 2025                  # 只抓 2025 年
        """
    )
    parser.add_argument("--source", choices=["arxiv", "openalex"], default="openalex",
                        help="数据源 (默认: openalex)")
    parser.add_argument("--full", action="store_true",
                        help="全量抓取所有年份")
    parser.add_argument("--year", type=int, default=None,
                        help="只抓取指定年份")
    parser.add_argument("--years", type=str, default=None,
                        help="抓取多个年份，逗号分隔 (如: 2023,2024,2025)")
    parser.add_argument("--update-keywords", action="store_true",
                        help="抓取前先用 LLM 更新关键词")
    args = parser.parse_args()

    # 更新关键词（可选）
    if args.update_keywords:
        logger.info("Updating keywords with LLM...")
        from keyword_generator import update_keywords
        update_keywords()

    # 加载已有数据
    papers = load_existing_papers()
    existing_years = get_existing_years(papers)
    logger.info(f"Existing years: {sorted(existing_years)}")

    # 确定要抓取的年份
    if args.year:
        years = [args.year]
    elif args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    elif args.full:
        years = [y for y in range(START_YEAR, END_YEAR + 1) if y not in existing_years]
        if END_YEAR not in years:
            years.append(END_YEAR)
    else:
        years = None  # 增量模式

    incremental = years is None

    # 选择数据源并抓取
    if args.source == "openalex":
        logger.info(f"使用 OpenAlex API ({'增量' if incremental else '按年'}模式)")
        papers = fetch_openalex(papers, years, incremental)
    else:
        logger.info(f"使用 arXiv API ({'增量' if incremental else '按年'}模式)")
        papers = fetch_arxiv(papers, years, incremental)

    # 保存
    save_papers(papers)
    logger.info(f"Done! Total: {len(papers)} papers")


if __name__ == "__main__":
    main()
