"""
arXiv API scraper for Paper-Hub
"""

import time
import urllib.parse
import feedparser
import logging
import json
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"


class ArxivScraper:
    def __init__(self, delay: float = 15.0):  # 增加到 15 秒
        self.delay = delay
        self.progress_file = "output/progress.json"

    def search(self, query: str, start: int = 0, max_results: int = 100, retries: int = 5) -> list:
        """Search arXiv API with retry logic"""
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        for attempt in range(retries):
            try:
                logger.info(f"Fetching (attempt {attempt+1}): {url[:80]}...")
                response = feedparser.parse(url)

                # Check for rate limit (429) - handle missing status attribute
                status = getattr(response, 'status', 200)
                if status == 429:
                    wait_time = (attempt + 1) * 60  # 60, 120, 180, 240, 300 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                time.sleep(self.delay)

                papers = []
                for entry in response.entries:
                    paper = {
                        "id": entry.id.split("/")[-1],
                        "title": entry.title.replace("\n", " ").strip(),
                        "abstract": entry.summary.replace("\n", " ").strip(),
                        "authors": [a.name for a in entry.authors],
                        "published": entry.published.split("T")[0],
                        "updated": entry.updated.split("T")[0],
                        "categories": [tag.term for tag in entry.tags],
                        "pdf_url": None,
                        "arxiv_url": entry.id,
                    }

                    # Find PDF link
                    for link in entry.links:
                        if link.type == "application/pdf":
                            paper["pdf_url"] = link.href
                            break

                    papers.append(paper)

                return papers

            except Exception as e:
                logger.error(f"Error fetching: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                continue

        return []

    def fetch_all(self, query: str, max_total: int = 1000, batch_size: int = 100) -> list:
        """Fetch all papers matching query"""
        all_papers = []
        start = 0
        empty_count = 0  # 连续空结果计数
        max_empty = 2    # 最大连续空结果数

        while start < max_total:
            papers = self.search(query, start=start, max_results=batch_size)

            if not papers:
                empty_count += 1
                if empty_count >= max_empty:
                    logger.warning(f"Empty results for {max_empty} consecutive batches, stopping")
                    break
                # 单次空结果可能是临时问题，继续尝试下一批
                logger.warning(f"Empty result, retrying next batch (empty_count={empty_count})")
                start += batch_size
                continue

            empty_count = 0  # 重置计数
            all_papers.extend(papers)
            logger.info(f"Fetched {len(all_papers)} papers so far...")

            if len(papers) < batch_size:
                break

            start += batch_size

        logger.info(f"Total fetched: {len(all_papers)} papers")
        return all_papers

    def load_progress(self) -> dict:
        """Load progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"last_update": None, "papers": []}

    def save_progress(self, papers: list):
        """Save progress to file"""
        os.makedirs("output", exist_ok=True)
        progress = {
            "last_update": datetime.now().isoformat(),
            "papers": papers
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)