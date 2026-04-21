"""
Pipeline for cleaning, classifying, and exporting papers
"""

import argparse
import json
import csv
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from cleaning import clean_papers, deduplicate, get_statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_papers(input_file: str) -> list:
    """Load papers from JSON file"""
    with open(input_file, 'r') as f:
        papers = json.load(f)
    logger.info(f"Loaded {len(papers)} papers")
    return papers


def save_json(papers: list, output_file: str):
    """Save papers to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}")


def save_csv(papers: list, output_file: str):
    """Save papers to CSV"""
    if not papers:
        return

    fieldnames = [
        "id", "title", "authors", "published", "year",
        "categories", "arxiv_url", "pdf_url", "code",
        "_domains", "_tasks", "publication"
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            row = {k: paper.get(k, "") for k in fieldnames}

            # Convert lists to strings
            if isinstance(row["authors"], list):
                row["authors"] = "; ".join(row["authors"])
            if isinstance(row["categories"], list):
                row["categories"] = "; ".join(row["categories"])
            if isinstance(row["_domains"], list):
                row["_domains"] = "; ".join(row["_domains"])
            if isinstance(row["_tasks"], list):
                row["_tasks"] = "; ".join(row["_tasks"])

            writer.writerow(row)

    logger.info(f"Saved to {output_file}")


def export_by_domain(papers: list, output_dir: str):
    """Export papers by domain"""
    domain_files = {
        "world_model": "papers_world_model.json",
        "physical_ai": "papers_physical_ai.json",
        "medical_ai": "papers_medical_ai.json"
    }

    for domain, filename in domain_files.items():
        domain_papers = [p for p in papers if domain in p.get("_domains", [])]
        if domain_papers:
            save_json(domain_papers, f"{output_dir}/{filename}")
            save_csv(domain_papers, f"{output_dir}/{filename.replace('.json', '.csv')}")
            logger.info(f"Exported {len(domain_papers)} papers for {domain}")


def generate_rss(papers: list, output_dir: str):
    """Generate RSS feed for recent papers"""
    # Sort by date and get last 7 days
    recent = sorted(papers, key=lambda x: x.get("published", ""), reverse=True)[:50]

    rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
  <title>Paper-Hub: World Model, Physical AI, Medical AI</title>
  <link>https://your-site.github.io/paper-hub</link>
  <description>Latest papers on World Model, Physical AI, and Medical AI</description>
  <language>en-us</language>
  <lastBuildDate>{build_date}</lastBuildDate>
  <atom:link href="https://your-site.github.io/paper-hub/output/feed.xml" rel="self" type="application/rss+xml"/>
""".format(build_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"))

    for paper in recent:
        rss += f"""  <item>
    <title><![CDATA[{paper.get('title', '')}]]></title>
    <link>{paper.get('arxiv_url', '')}</link>
    <description><![CDATA[{paper.get('abstract', '')}]]></description>
    <pubDate>{paper.get('published', '')}</pubDate>
    <guid>{paper.get('arxiv_url', '')}</guid>
  </item>
"""

    rss += """</channel>
</rss>"""

    with open(f"{output_dir}/feed.xml", 'w', encoding='utf-8') as f:
        f.write(rss)

    logger.info(f"Generated RSS feed with {len(recent)} items")


def main():
    parser = argparse.ArgumentParser(description="Process papers")
    parser.add_argument("--input", default="output/papers_raw.json")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    # Load raw papers - check progress.json first, then papers_raw.json
    input_file = args.input
    if not os.path.exists(input_file):
        alt_file = "output/progress.json"
        if os.path.exists(alt_file):
            logger.info(f"Loading from {alt_file}")
            with open(alt_file, 'r') as f:
                data = json.load(f)
                papers = data.get("papers", [])
        else:
            raise FileNotFoundError(f"No input file found: {input_file}")
    else:
        papers = load_papers(input_file)

    # Deduplicate
    papers = deduplicate(papers)
    logger.info(f"After dedup: {len(papers)} papers")

    # Clean and classify
    papers = clean_papers(papers)
    logger.info(f"After domain过滤: {len(papers)} papers")

    # Get statistics
    stats = get_statistics(papers)
    logger.info(f"Statistics: {stats}")

    # Save main output
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(papers, f"{args.output_dir}/papers.json")
    save_csv(papers, f"{args.output_dir}/papers.csv")

    # Export by domain
    export_by_domain(papers, args.output_dir)

    # Generate RSS
    generate_rss(papers, args.output_dir)

    # Save statistics
    with open(f"{args.output_dir}/statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("Pipeline complete!")
    logger.info(f"Total papers: {stats['total']}")
    for domain, count in stats.get('domains', {}).items():
        logger.info(f"  {domain}: {count}")


if __name__ == "__main__":
    main()