"""
Main entry point for paper fetching
- 使用多个搜索查询以获取更多论文
- 默认并行抓取所有年份
- 自动增量更新（只抓取缺失的年份）
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SEARCH_QUERIES, START_YEAR, MAX_RESULTS
from config_search import ArxivScraper

# 动态获取当前年份
END_YEAR = datetime.now().year
OUTPUT_DIR = "output"
PARALLEL_WORKERS = 3  # 并行抓取年份数

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_query_year(args_tuple):
    """Fetch papers for a single query + year"""
    query, year, max_results, delay = args_tuple
    scraper = ArxivScraper(delay=delay)
    full_query = f"{query} AND submittedDate:[{year}0101 TO {year}1231]"
    papers = scraper.fetch_all(full_query, max_total=max_results)
    return query[:30], year, papers


def load_existing_papers():
    """Load existing papers from previous runs"""
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")

    existing = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                for p in data.get("papers", []):
                    existing[p["id"]] = p
            logger.info(f"Loaded {len(existing)} existing papers")
        except Exception as e:
            logger.warning(f"Could not load progress.json: {e}")

    return existing


def save_papers(papers):
    """Save papers to progress.json"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    progress = {
        "last_update": datetime.now().isoformat(),
        "papers": papers
    }
    with open(os.path.join(OUTPUT_DIR, "progress.json"), 'w') as f:
        json.dump(progress, f, indent=2)

    logger.info(f"Saved {len(papers)} papers")


def main():
    base_queries = SEARCH_QUERIES

    # 加载已有论文
    existing_papers = load_existing_papers()

    # 检查已有论文的年份
    existing_years = set()
    for p in existing_papers.values():
        if 'published' in p:
            try:
                year = int(p['published'][:4])
                existing_years.add(year)
            except:
                pass

    # 需要抓取的年份（2023-2026，缺失的年份）
    all_years = list(range(START_YEAR, END_YEAR + 1))
    years_to_fetch = [y for y in all_years if y not in existing_years]

    if not years_to_fetch:
        logger.info(f"All years already fetched. Total papers: {len(existing_papers)}")
        return list(existing_papers.values())

    logger.info(f"Fetching years: {years_to_fetch}")
    logger.info(f"Using {len(base_queries)} search queries per year")

    # 合并已有论文和新抓取的论文
    all_papers = existing_papers.copy()

    # 为每个查询+年份组合创建任务
    # 先按年份分组，每年内并行执行多个查询
    for year in years_to_fetch:
        logger.info(f"=== Fetching year {year} ===")
        tasks = [(query, year, 1000, 3.0) for query in base_queries]
        year_papers = {}

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_query_year, task): task[0] for task in tasks}

            for future in as_completed(futures):
                query_short, _, papers = future.result()
                if papers:
                    new_count = 0
                    for p in papers:
                        if p['id'] not in all_papers:
                            all_papers[p['id']] = p
                            year_papers[p['id']] = p
                            new_count += 1
                    logger.info(f"  Query '{query_short}...': {len(papers)} papers ({new_count} new)")
                else:
                    logger.info(f"  Query '{query_short}...': 0 papers")

        logger.info(f"Year {year} done: got {len(year_papers)} new papers")

    # 保存
    papers_list = list(all_papers.values())
    save_papers(papers_list)

    logger.info(f"Done! Total: {len(papers_list)} papers")
    return papers_list


if __name__ == "__main__":
    main()