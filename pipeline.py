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
from config import TASK_DEFINITIONS, TASK_EN_LABELS, TASK_DOMAIN_MAP

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


def export_task_meta(output_dir: str):
    """Export task metadata (labels + domain mapping) for dynamic frontend rendering."""
    tasks = {}
    domain_tasks: dict = {"world_model": [], "physical_ai": [], "medical_ai": []}

    for abbr, (zh_label, _keywords) in TASK_DEFINITIONS.items():
        en_label = TASK_EN_LABELS.get(abbr, abbr)
        tasks[abbr] = {"zh": zh_label, "en": en_label}

        domain = TASK_DOMAIN_MAP.get(abbr)
        if domain and domain in domain_tasks:
            domain_tasks[domain].append(abbr)

    meta = {"tasks": tasks, "domain_tasks": domain_tasks}
    out_path = f"{output_dir}/task_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Exported task_meta.json: {len(tasks)} tasks across {sum(len(v) for v in domain_tasks.values())} domain slots")


def compute_trending(papers: list, output_dir: str, months: int = 6, top_n: int = 8):
    """
    Extract truly trending phrases from recent paper titles+abstracts per domain.
    Uses domain-specificity scoring so generic ML terms (e.g. 'latent space') are
    ranked below phrases that are distinctive to a particular domain.
    """
    import re
    from collections import Counter
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=months * 30)).isoformat()
    recent = [p for p in papers if p.get("published", "") >= cutoff]
    if not recent:
        recent = sorted(papers, key=lambda x: x.get("published", ""), reverse=True)[:800]

    STOP = {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "this", "that", "is", "are", "was", "were",
        "be", "been", "have", "has", "do", "does", "we", "our", "it", "its",
        "they", "which", "as", "such", "can", "also", "not", "but", "than",
        "based", "proposed", "approach", "method", "model", "models", "paper",
        "propose", "presents", "present", "show", "shows", "use", "using",
        "used", "results", "demonstrate", "work", "however", "state", "art",
        "two", "three", "new", "novel", "existing", "recent", "learning",
        "deep", "neural", "network", "networks", "data", "task", "tasks",
        "training", "trained", "large", "high", "low", "via", "into",
        "each", "both", "across", "while", "without", "further", "thus",
        "extensive", "experiments", "outperforms", "significantly", "achieves",
        "benchmark", "performance", "evaluation", "superior", "experimental",
        "https", "github", "http", "com", "available", "code", "page",
        "project", "arxiv", "www",
        # Abbreviations that create noisy duplicate phrases
        "pdes", "pinns", "pinn", "odes",
    }
    # Plural/variant normalization for dedup (not for display)
    NORM = {
        "images": "image", "models": "model", "networks": "network",
        "equations": "equation", "methods": "method", "agents": "agent",
        "fields": "field", "operators": "operator", "algorithms": "algorithm",
        "systems": "system", "problems": "problem", "tasks": "task",
    }
    BOILERPLATE = re.compile(
        r"(extensive experiment|state.of.the.art|code available|project page"
        r"|success rate|real.world|significantly outperform|achieves state"
        r"|available https|github com|page https)",
        re.IGNORECASE,
    )
    # Known acronyms to display in uppercase
    UPPER_WORDS = {"pinn", "fno", "vla", "rl", "nlp", "mri", "ct", "vae",
                   "gan", "llm", "vlm", "nerf", "ai", "3d", "2d", "ood",
                   "cnn", "rnn", "gnn", "gpt", "ehr", "wsi", "oct"}

    def _display(term: str) -> str:
        parts = []
        for w in term.split():
            parts.append(w.upper() if w in UPPER_WORDS else w.capitalize())
        return " ".join(parts)

    def _tokenize(text: str):
        tokens = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", text.lower())
        return [t for t in tokens if t not in STOP and len(t) > 2]

    def _ngrams(texts):
        counter: Counter = Counter()
        for text in texts:
            toks = _tokenize(text)
            for n in (2, 3):
                for i in range(len(toks) - n + 1):
                    counter[" ".join(toks[i: i + n])] += 1
        return counter

    domains = ["world_model", "physical_ai", "medical_ai"]
    # Title weighted 2× to surface concise topic phrases
    domain_texts = {
        d: [f"{p.get('title','')} {p.get('title','')} {p.get('abstract','')[:400]}"
            for p in recent if d in p.get("_domains", [])]
        for d in domains
    }
    domain_counts = {d: max(len(domain_texts[d]), 1) for d in domains}
    domain_ngrams = {d: _ngrams(domain_texts[d]) for d in domains}

    all_texts = [t for ts in domain_texts.values() for t in ts]
    all_ngrams = _ngrams(all_texts)
    total = max(len(all_texts), 1)

    result = {}
    for domain in domains:
        n_papers = domain_counts[domain]
        candidates = []
        for gram, freq in domain_ngrams[domain].most_common(1000):
            if freq < 3:
                break
            if BOILERPLATE.search(gram):
                continue
            global_freq = all_ngrams.get(gram, 0)
            # Specificity: domain rate vs overall rate
            specificity = (freq / n_papers) / (global_freq / total + 1e-6)
            if specificity > 1.3:
                candidates.append({
                    "term": gram,
                    "display": _display(gram),
                    "count": freq,
                    "specificity": round(specificity, 2),
                })
        candidates.sort(key=lambda x: -x["count"])

        # Deduplicate: prefer longer (more specific) phrases over shorter fragments.
        # If a new phrase is a STRICT SUPERSET of an already-selected one, replace it.
        # If a new phrase is a STRICT SUBSET of an already-selected one, skip it.
        def _tok_set(term: str):
            # Normalize plurals/variants so 'images' == 'image' for dedup purposes
            return {NORM.get(w, w) for w in re.split(r"[\s\-]+", term.lower())}

        deduped: list = []
        for c in candidates:
            c_words = _tok_set(c["term"])
            skip = False
            replacements = []     # indices in deduped to replace with c

            for idx, existing in enumerate(deduped):
                e_words = _tok_set(existing["term"])
                if c_words == e_words:
                    skip = True; break          # exact duplicate
                elif c_words > e_words:
                    replacements.append(idx)    # c is more specific → replace existing
                elif c_words < e_words:
                    skip = True; break          # existing is more specific → skip c

            if skip:
                continue
            if replacements:
                # Replace all subsumed shorter phrases with c (keep first slot)
                for idx in sorted(replacements, reverse=True):
                    deduped.pop(idx)
            deduped.append(c)
            if len(deduped) >= top_n:
                break

        result[domain] = deduped

    out = {
        "generated": datetime.now().isoformat()[:10],
        "months": months,
        "trends": result,
    }
    with open(f"{output_dir}/trending.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(
        "Computed trending: " + ", ".join(f"{d}:{len(result[d])}" for d in domains)
    )


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

    # Export task metadata for dynamic frontend rendering
    export_task_meta(args.output_dir)

    # Compute trending topics from recent paper abstracts
    compute_trending(papers, args.output_dir)

    # Save statistics
    with open(f"{args.output_dir}/statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("Pipeline complete!")
    logger.info(f"Total papers: {stats['total']}")
    for domain, count in stats.get('domains', {}).items():
        logger.info(f"  {domain}: {count}")


if __name__ == "__main__":
    main()