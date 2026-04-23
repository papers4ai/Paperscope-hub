"""
OpenAlex API scraper for Paper-Hub
- 免费，限制宽松（每秒 10 个请求）
- 覆盖 arXiv 论文
- 提供更丰富的元数据

OpenAlex API 文档: https://docs.openalex.org/
"""

import time
import requests
import logging
import re
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

OPENALEX_API_URL = "https://api.openalex.org"

# arXiv source ID in OpenAlex
ARXIV_SOURCE_ID = "S4306400194"

# 可选：注册邮箱可获得更高限制
# https://openalex.org/how-to-use-the-api/rate-limits-and-authentication
OPENALEX_EMAIL = None  # 设置您的邮箱可提高限制


class OpenAlexScraper:
    def __init__(self, delay: float = 0.5, email: str = None):
        self.delay = delay
        self.email = email or OPENALEX_EMAIL
        self.session = requests.Session()
        if self.email:
            self.session.headers.update({"mailto": self.email})

    def _make_request(self, endpoint: str, params: dict, retries: int = 5) -> dict:
        """Make API request with retry logic"""
        url = f"{OPENALEX_API_URL}/{endpoint}"

        for attempt in range(retries):
            try:
                logger.info(f"Fetching: {url}")
                logger.debug(f"Params: {params}")
                response = self.session.get(url, params=params, timeout=60)

                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 400:
                    logger.error(f"Bad Request: {response.text}")
                    raise requests.exceptions.RequestException(f"400 Bad Request: {response.text}")

                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                raise

        return {}

    def search_by_keyword(self, keyword: str, from_date: str = None, to_date: str = None,
                          max_results: int = 500, per_page: int = 200) -> list:
        """
        Search papers by keyword in title

        Args:
            keyword: Search keyword
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_results: Maximum total results
            per_page: Results per page (max 200)
        """
        all_papers = []
        page = 1
        total_fetched = 0

        # Build filter string
        filter_parts = []

        # Title search
        filter_parts.append(f'title.search:"{keyword}"')

        # Date range
        if from_date or to_date:
            filter_parts.append(f"from_publication_date:{from_date or '1900-01-01'}")
            if to_date:
                filter_parts.append(f"to_publication_date:{to_date}")

        # arXiv source filter
        filter_parts.append(f"primary_location.source.id:{ARXIV_SOURCE_ID}")

        params = {
            "filter": ",".join(filter_parts),
            "per_page": min(per_page, 200),
            "sort": "publication_date:desc",
            "select": "id,doi,title,abstract_inverted_index,authorships,publication_date,primary_location,open_access"
        }

        while total_fetched < max_results:
            params["page"] = page
            data = self._make_request("works", params)

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                paper = self._parse_paper(item)
                if paper:
                    all_papers.append(paper)

            total_fetched += len(results)
            logger.info(f"Fetched {total_fetched} papers for '{keyword}'...")

            # Check if more pages available
            meta = data.get("meta", {})
            if total_fetched >= meta.get("count", 0):
                break

            page += 1

        logger.info(f"Total fetched for '{keyword}': {len(all_papers)} papers")
        return all_papers

    def search_papers(self, query: str, from_date: str = None, to_date: str = None,
                      max_results: int = 1000, per_page: int = 200) -> list:
        """
        Search papers - accepts keyword string

        Args:
            query: Search keyword
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_results: Maximum total results
            per_page: Results per page (max 200)
        """
        # Extract keyword from query if formatted
        keyword = query
        if query.startswith('title.search:'):
            match = re.search(r'"([^"]+)"', query)
            if match:
                keyword = match.group(1)

        return self.search_by_keyword(keyword, from_date, to_date, max_results, per_page)

    def _parse_paper(self, item: dict) -> dict:
        """Parse OpenAlex work item to our format"""
        if not item:
            return None

        try:
            # Extract arXiv ID from DOI or URL
            arxiv_id = None
            doi = item.get("doi", "") or ""
            if doi and "arxiv" in doi.lower():
                # DOI format: 10.48550/arXiv.2301.00001
                arxiv_id = doi.split("arXiv.")[-1] if "arXiv." in doi else doi.split("arxiv.")[-1]

            # Also try to get from OpenAlex ID
            if not arxiv_id:
                openalex_id = item.get("id", "")
                # OpenAlex ID format: https://openalex.org/W1234567890
                # We'll use the OpenAlex ID as fallback
                arxiv_id = openalex_id.split("/")[-1]

            # Get abstract
            abstract_inverted = item.get("abstract_inverted_index", {})
            abstract = self._reconstruct_abstract(abstract_inverted) if abstract_inverted else ""

            # Get authors
            authors = []
            for a in item.get("authorships", []):
                author = a.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            # Get PDF URL
            pdf_url = None
            location = item.get("primary_location") or {}
            if location:
                pdf_url = location.get("pdf_url")
                if not pdf_url:
                    landing = location.get("landing_page_url", "")
                    if "arxiv.org" in landing:
                        pdf_url = landing.replace("/abs/", "/pdf/") + ".pdf"

            open_access = item.get("open_access") or {}
            if not pdf_url and open_access.get("oa_url"):
                pdf_url = open_access["oa_url"]

            # Get publication date
            pub_date = item.get("publication_date", "")

            # Construct arXiv URL if we have an arXiv ID
            arxiv_url = None
            if arxiv_id and (doi and "arxiv" in doi.lower()):
                # Clean arXiv ID (remove version if present)
                clean_id = arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "")
                arxiv_url = f"https://arxiv.org/abs/{clean_id}"

            return {
                "id": arxiv_id,
                "title": item.get("title", "") or "",
                "abstract": abstract,
                "authors": authors,
                "published": pub_date,
                "updated": pub_date,
                "categories": [],
                "pdf_url": pdf_url,
                "arxiv_url": arxiv_url,
                "doi": doi,
                "openalex_id": item.get("id"),
            }
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            return None

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from inverted index"""
        if not inverted_index:
            return ""

        positions = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions.append((pos, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positions)

    def fetch_by_keywords(self, keywords: list, from_date: str = None, to_date: str = None,
                          max_results: int = 500) -> list:
        """
        Fetch papers by keywords

        Args:
            keywords: List of keywords to search
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_results: Max results per keyword
        """
        all_papers = {}
        seen_ids = set()

        for kw in keywords:
            logger.info(f"Searching keyword: {kw}")

            papers = self.search_by_keyword(
                keyword=kw,
                from_date=from_date,
                to_date=to_date,
                max_results=max_results
            )

            for p in papers:
                if p["id"] not in seen_ids:
                    all_papers[p["id"]] = p
                    seen_ids.add(p["id"])

            logger.info(f"Keyword '{kw}': {len(papers)} papers, total unique: {len(all_papers)}")

        return list(all_papers.values())
