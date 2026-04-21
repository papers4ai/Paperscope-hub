"""
Cleaning and Domain Classification Module
"""

import re
from typing import Dict, List, Tuple
from config import (
    WORLD_MODEL_KEYWORDS,
    PHYSICAL_AI_KEYWORDS,
    MEDICAL_AI_KEYWORDS,
    TASK_DEFINITIONS,
    MODEL_KEYWORDS
)


def load_keywords():
    """Load all keywords"""
    return {
        "world_model": WORLD_MODEL_KEYWORDS,
        "physical_ai": PHYSICAL_AI_KEYWORDS,
        "medical_ai": MEDICAL_AI_KEYWORDS
    }


def compile_patterns(keywords: List[str]) -> List:
    """Compile keyword patterns"""
    return [re.compile(kw, re.IGNORECASE) for kw in keywords]


# Pre-compile all patterns
_PATTERNS = {
    "world_model": compile_patterns(WORLD_MODEL_KEYWORDS),
    "physical_ai": compile_patterns(PHYSICAL_AI_KEYWORDS),
    "medical_ai": compile_patterns(MEDICAL_AI_KEYWORDS)
}

_TASK_PATTERNS = {
    abbr: [re.compile(kw, re.IGNORECASE) for kw in keywords]
    for abbr, (_, keywords) in TASK_DEFINITIONS.items()
}


def extract_year(published: str) -> int:
    """Extract year from published date"""
    try:
        return int(published[:4])
    except:
        return 2020


def clean_paper(paper: dict) -> dict:
    """Clean and normalize paper data"""
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", "").strip(),
        "abstract": paper.get("abstract", "").strip(),
        "authors": paper.get("authors", [])[:10],  # Limit to 10 authors
        "published": paper.get("published", ""),
        "year": extract_year(paper.get("published", "")),
        "categories": paper.get("categories", []),
        "pdf_url": paper.get("pdf_url", ""),
        "arxiv_url": paper.get("arxiv_url", ""),
    }


def check_domain(title: str, abstract: str, domain: str) -> Tuple[bool, List[str]]:
    """Check if paper belongs to a domain and return matched keywords"""
    text = f"{title} {abstract}"
    patterns = _PATTERNS.get(domain, [])

    matched = []
    for pattern in patterns:
        if pattern.search(text):
            matched.append(pattern.pattern)

    return len(matched) > 0, matched


def classify_domains(paper: dict) -> dict:
    """Classify paper into domains"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    domains = []
    domain_keywords = {}

    for domain in ["world_model", "physical_ai", "medical_ai"]:
        is_match, keywords = check_domain(title, abstract, domain)
        if is_match:
            domains.append(domain)
            domain_keywords[domain] = keywords

    paper["_domains"] = domains
    paper["_domain_keywords"] = domain_keywords

    return paper


def tag_tasks(paper: dict) -> dict:
    """Tag paper with task labels"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract}"

    matched_tasks = []
    task_details = {}

    for abbr, patterns in _TASK_PATTERNS.items():
        keywords = []
        for pattern in patterns:
            if pattern.search(text):
                keywords.append(pattern.pattern)

        if keywords:
            matched_tasks.append(abbr)
            task_details[abbr] = keywords

    paper["_tasks"] = matched_tasks
    paper["_task_details"] = task_details

    return paper


def extract_code_links(text: str) -> List[str]:
    """Extract code repository links from text (abstract + title)"""
    links = []
    patterns = [
        # GitHub (use negative lookahead to exclude trailing punctuation)
        r"(https?://github\.com/[a-zA-Z0-9][-a-zA-Z0-9]*/[a-zA-Z0-9][-._a-zA-Z0-9]*)",
        # GitLab
        r"(https?://gitlab\.com/[a-zA-Z0-9][-a-zA-Z0-9]*/[a-zA-Z0-9][-._a-zA-Z0-9]*)",
        # Gitee
        r"(https?://gitee\.com/[a-zA-Z0-9][-a-zA-Z0-9]*/[a-zA-Z0-9][-._a-zA-Z0-9]*)",
        # HuggingFace
        r"(https?://huggingface\.co/[a-zA-Z0-9][-a-zA-Z0-9]*)",
        # Bitbucket
        r"(https?://bitbucket\.org/[a-zA-Z0-9][-a-zA-Z0-9]*/[a-zA-Z0-9][-._a-zA-Z0-9]*)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        links.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    cleaned = []
    for link in links:
        link = link.rstrip('.,;:!?。；：！？')
        # Ensure each unique link appears only once
        if link not in seen and len(link) > 10:
            seen.add(link)
            cleaned.append(link)

    return cleaned

    return list(set(cleaned))[:3]  # Max 3 links


def extract_month(published: str) -> int:
    """Extract month from published date"""
    try:
        return int(published[5:7])
    except:
        return 1


def classify_paper_type(title: str, abstract: str) -> str:
    """Classify paper type: Method, Dataset, or Survey"""
    text = f"{title} {abstract}"

    # Survey patterns (highest priority)
    survey_patterns = [
        r"\bsurvey\b", r"\breview\b", r"\boverview\b",
        r"\btutorial\b", r"\bguided\b",
        r"comprehensive\s+(study|analysis|review|survey)",
        r"systematic\s+review", r"literature\s+review",
    ]
    for p in survey_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Survey"

    # Dataset patterns
    dataset_patterns = [
        r"\bdataset\b", r"\bbenchmark\b", r"\bcorpus\b",
        r"introduce\s+(a\s+)?(new\s+)?(large[-\s]?scale\s+)?dataset",
        r"present\s+(a\s+)?(new\s+)?dataset",
        r"release\s+(a\s+)?(new\s+)?(large\s+)?dataset",
        r"construct\s+(a\s+)?(new\s+)?dataset",
    ]
    for p in dataset_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Dataset"

    # Default to Method
    return "Method"


def extract_publication(title: str) -> str:
    """Extract publication venue from title (heuristic)"""
    # Common venue patterns
    venues = [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV",
        "AAAI", "IJCAI", "ACL", "EMNLP", "NAACL",
        "Nature", "Science", "Cell", "Lancet"
    ]

    for venue in venues:
        if venue.lower() in title.lower():
            return venue

    return ""


def clean_papers(papers: list) -> list:
    """Clean and annotate all papers"""
    cleaned = []

    for paper in papers:
        # Clean basic fields
        paper = clean_paper(paper)

        # Extract code links (search in both title and abstract)
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        code_links = extract_code_links(text)
        paper["code"] = code_links[0] if code_links else ""
        paper["has_code"] = len(code_links) > 0  # Flag for frontend

        # Extract publication venue
        paper["publication"] = extract_publication(paper.get("title", ""))

        # Extract month
        paper["month"] = extract_month(paper.get("published", ""))

        # Classify paper type (Method/Dataset/Survey)
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        paper["type"] = classify_paper_type(title, abstract)

        # Classify domains
        paper = classify_domains(paper)

        # Tag tasks
        paper = tag_tasks(paper)

        # Keep all papers (no filtering) - search query already filters relevant papers
        cleaned.append(paper)

    return cleaned


def deduplicate(papers: list) -> list:
    """Remove duplicate papers by arxiv URL"""
    seen = set()
    unique = []

    for paper in papers:
        url = paper.get("arxiv_url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(paper)

    return unique


def get_statistics(papers: list) -> dict:
    """Get statistics about papers"""
    from collections import Counter

    domain_count = Counter()
    task_count = Counter()
    year_count = Counter()

    for paper in papers:
        for domain in paper.get("_domains", []):
            domain_count[domain] += 1

        for task in paper.get("_tasks", []):
            task_count[task] += 1

        year_count[paper.get("year", 2020)] += 1

    return {
        "total": len(papers),
        "domains": dict(domain_count),
        "tasks": dict(task_count),
        "years": dict(year_count)
    }