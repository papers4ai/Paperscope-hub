"""
Dynamic keyword optimizer: mines paper abstracts to find missing keywords.

Usage:
  python keyword_optimizer.py                    # analyze all domains
  python keyword_optimizer.py --domain medical_ai
  python keyword_optimizer.py --top 40 --min-freq 4
  python keyword_optimizer.py --apply            # patch config.py (requires LLM)

Flow:
  1. Load output/papers.json
  2. Per domain: find "uncovered" papers (_tasks is empty)
  3. Extract high-frequency n-grams from title+abstract
  4. Filter out phrases already matched by existing keyword patterns
  5. LLM groups top candidates and suggests TASK_DEFINITIONS additions
  6. Print report; optionally patch config.py with --apply
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from config import TASK_DEFINITIONS, WORLD_MODEL_KEYWORDS, PHYSICAL_AI_KEYWORDS, MEDICAL_AI_KEYWORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── stopwords ────────────────────────────────────────────────────────────────
_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "we", "our", "us", "i", "it", "its", "they", "their", "which", "who",
    "as", "such", "can", "also", "not", "no", "but", "than", "more",
    "based", "proposed", "approach", "method", "model", "models", "paper",
    "propose", "presents", "present", "show", "shows", "use", "using",
    "used", "results", "demonstrate", "demonstrates", "work", "however",
    "state", "art", "two", "three", "new", "novel", "existing", "recent",
    "learning", "deep", "neural", "network", "networks", "data", "task",
    "tasks", "training", "trained", "large", "high", "low", "via", "into",
    "each", "both", "across", "while", "without", "further", "thus", "well",
    # boilerplate phrases (filtered at n-gram level too)
    "extensive", "experiments", "outperforms", "significantly", "achieves",
    "state-of-the-art", "benchmark", "performance", "evaluation", "superior",
    "experimental", "ablation", "baseline", "baselines", "metrics",
    # URL fragments
    "https", "github", "http", "com", "available", "code", "page", "project",
    "arxiv", "www", "html", "pdf",
}

# boilerplate multi-word phrases to reject regardless of frequency
_BOILERPLATE_RE = re.compile(
    r"(extensive experiment|state.of.the.art|code available|project page"
    r"|success rate|real.world|significantly outperform|achieves state"
    r"|world world|available https|github com|page https)",
    re.IGNORECASE,
)

# ── domain → task subset mapping (used to judge if a task is relevant) ───────
_DOMAIN_TASKS = {
    "world_model":  {"VidGen", "NeRF", "MBRL", "Sim2Real", "EmbodiedWM", "Predictive"},
    "physical_ai":  {"PINN", "NeuralOp", "Embodied", "RobotLearn", "FluidSim", "Climate", "3DRecon"},
    "medical_ai":   {"Pathology", "MedImg", "Cancer", "MedVLM", "DrugMol", "Protein", "Clinical",
                     "Surgery", "HealthMon"},
}

# Pre-compile all existing keyword patterns (across all tasks)
_ALL_PATTERNS: List[re.Pattern] = [
    re.compile(kw, re.IGNORECASE)
    for _, kws in TASK_DEFINITIONS.values()
    for kw in kws
]
# Also include domain-level patterns
for kw in WORLD_MODEL_KEYWORDS + PHYSICAL_AI_KEYWORDS + MEDICAL_AI_KEYWORDS:
    try:
        _ALL_PATTERNS.append(re.compile(kw, re.IGNORECASE))
    except re.error:
        pass


# ── n-gram extraction ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 2]


def _extract_ngrams(texts: List[str], n_range=(2, 3)) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        tokens = _tokenize(text)
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i: i + n])
                counter[gram] += 1
    return counter


def _is_covered(phrase: str) -> bool:
    """Return True if phrase is already matched by any existing keyword pattern."""
    for pat in _ALL_PATTERNS:
        if pat.search(phrase):
            return True
    return False


def _uncovered_ngrams(
    papers: List[Dict], min_freq: int, top_n: int
) -> List[Tuple[str, int]]:
    texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
    counts = _extract_ngrams(texts)

    candidates = [
        (gram, freq)
        for gram, freq in counts.most_common(5000)
        if freq >= min_freq
        and not _is_covered(gram)
        and not _BOILERPLATE_RE.search(gram)
    ]
    return candidates[:top_n]


# ── load papers ───────────────────────────────────────────────────────────────

def load_papers(path: str = "output/papers.json") -> List[Dict]:
    if not os.path.exists(path):
        alt = "output/progress.json"
        if os.path.exists(alt):
            with open(alt) as f:
                return json.load(f).get("papers", [])
        raise FileNotFoundError(f"No papers file found at {path}")
    with open(path) as f:
        return json.load(f)


# ── per-domain analysis ───────────────────────────────────────────────────────

def analyze_domain(
    domain: str,
    all_papers: List[Dict],
    min_freq: int,
    top_n: int,
) -> Dict:
    domain_papers = [p for p in all_papers if domain in p.get("_domains", [])]
    if not domain_papers:
        return {"domain": domain, "total": 0, "uncovered": 0, "candidates": []}

    relevant_tasks = _DOMAIN_TASKS.get(domain, set())
    uncovered = [
        p for p in domain_papers
        if not any(t in relevant_tasks for t in p.get("_tasks", []))
    ]

    logger.info(
        f"[{domain}] {len(domain_papers)} papers total, "
        f"{len(uncovered)} without task labels ({len(uncovered)/len(domain_papers):.0%})"
    )

    candidates = _uncovered_ngrams(uncovered, min_freq=min_freq, top_n=top_n)
    return {
        "domain": domain,
        "total": len(domain_papers),
        "uncovered": len(uncovered),
        "candidates": candidates,
    }


# ── LLM suggestion ────────────────────────────────────────────────────────────

_SUGGEST_PROMPT = """\
You are an AI research keyword curator. Below is a domain and a list of n-grams \
(phrases) extracted from paper abstracts in that domain. These phrases are NOT yet \
covered by the existing keyword taxonomy.

Domain: {domain}
Existing task subtopics (already covered): {existing_tasks}

Top uncovered phrases (phrase → frequency):
{phrases}

Your job:
1. Identify which phrases represent genuine research subtopics worth tracking.
2. Group related phrases into 2-6 proposed new TASK entries (or extensions to existing ones).
3. For each group, provide:
   - short_name: 6-char uppercase abbreviation (e.g. "DifPol")
   - label: Chinese label (e.g. "扩散策略")
   - action: "new" | "extend:<ExistingTaskName>"
   - keywords: list of regex patterns to add (2-5 patterns, use \\s* for spaces)
   - reason: one sentence explaining why this is worth tracking

Return ONLY valid JSON, no explanation:
[
  {{
    "short_name": "DifPol",
    "label": "扩散策略",
    "action": "extend:RobotLearn",
    "keywords": ["diffusion\\\\s*policy", "action\\\\s*diffusion", "behavior\\\\s*cloning\\\\s*diffusion"],
    "reason": "Diffusion Policy is the dominant paradigm for robot manipulation learning since 2023."
  }}
]"""


def llm_suggest(domain_result: Dict, client, model: str) -> List[Dict]:
    domain = domain_result["domain"]
    candidates = domain_result["candidates"][:60]
    if not candidates:
        return []

    existing_tasks = list(_DOMAIN_TASKS.get(domain, set()))
    phrases_str = "\n".join(f"  {gram} → {freq}" for gram, freq in candidates)

    prompt = _SUGGEST_PROMPT.format(
        domain=domain,
        existing_tasks=", ".join(existing_tasks),
        phrases=phrases_str,
    )

    resp = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.split("```")[0].strip()
    return json.loads(text)


# ── config.py patcher ─────────────────────────────────────────────────────────

def _patch_config(suggestions_by_domain: Dict[str, List[Dict]]):
    """Append new/extended keywords to config.py TASK_DEFINITIONS."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path) as f:
        source = f.read()

    additions = []  # lines to append

    for domain, suggestions in suggestions_by_domain.items():
        for s in suggestions:
            action = s.get("action", "new")
            short_name = s["short_name"]
            label = s["label"]
            kws = s.get("keywords", [])
            if not kws:
                continue

            if action.startswith("extend:"):
                existing_task = action.split(":")[1]
                # Find the task in source and append keywords
                pattern = rf'("{existing_task}":\s*\("[^"]*",\s*\[)([\s\S]*?)(\])'
                match = re.search(pattern, source)
                if match:
                    kw_block = match.group(2)
                    new_kws = "".join(f'\n        r"{kw}",' for kw in kws)
                    new_block = match.group(1) + kw_block + new_kws + "\n    " + match.group(3)
                    source = source[: match.start()] + new_block + source[match.end():]
                    logger.info(f"Extended {existing_task} with {len(kws)} keywords")
                else:
                    logger.warning(f"Could not find task {existing_task} to extend — adding as new")
                    action = "new"

            if action == "new":
                kw_repr = ",\n        ".join(f'r"{kw}"' for kw in kws)
                additions.append(
                    f'    "{short_name}": ("{label}", [\n        {kw_repr}\n    ]),'
                )

    if additions:
        # Insert before closing brace of TASK_DEFINITIONS
        insert_point = source.rfind("\n}")
        block = "\n    # ── auto-added by keyword_optimizer ──\n" + "\n".join(additions)
        source = source[:insert_point] + block + source[insert_point:]

    with open(config_path, "w") as f:
        f.write(source)

    logger.info("config.py patched successfully")


# ── pretty-print report ───────────────────────────────────────────────────────

def _print_report(domain_result: Dict, suggestions: List[Dict]):
    domain = domain_result["domain"]
    print(f"\n{'='*60}")
    print(f"  Domain: {domain}")
    print(f"  Papers: {domain_result['total']}  |  Uncovered: {domain_result['uncovered']}")
    print(f"{'='*60}")

    print("\n  Top uncovered n-grams:")
    for gram, freq in domain_result["candidates"][:30]:
        print(f"    [{freq:>4}]  {gram}")

    if suggestions:
        print(f"\n  LLM suggestions ({len(suggestions)} groups):")
        for s in suggestions:
            action_str = s.get("action", "new")
            print(f"\n  [{s['short_name']}] {s['label']}  ({action_str})")
            print(f"    Reason: {s.get('reason', '')}")
            print(f"    Keywords: {s.get('keywords', [])}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keyword optimizer for Paperscope-hub")
    parser.add_argument("--domain", choices=["world_model", "physical_ai", "medical_ai"],
                        help="Analyze a single domain (default: all)")
    parser.add_argument("--top", type=int, default=50,
                        help="Max candidate phrases to show/send to LLM (default: 50)")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum phrase frequency to consider (default: 3)")
    parser.add_argument("--apply", action="store_true",
                        help="Patch config.py with LLM suggestions (requires LLM_API_KEY)")
    parser.add_argument("--input", default="output/papers.json",
                        help="Path to papers JSON (default: output/papers.json)")
    args = parser.parse_args()

    papers = load_papers(args.input)
    logger.info(f"Loaded {len(papers)} papers")

    domains = [args.domain] if args.domain else ["world_model", "physical_ai", "medical_ai"]

    # ── optional LLM client ──
    client = None
    model = None
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            base_url = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            model = os.environ.get("LLM_MODEL", "glm-4-flash")
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLM enabled: {model} @ {base_url}")
        except ImportError:
            logger.warning("openai package not installed — LLM suggestions disabled")
    else:
        logger.info("LLM_API_KEY not set — running frequency analysis only")

    suggestions_by_domain: Dict[str, List[Dict]] = {}

    for domain in domains:
        result = analyze_domain(domain, papers, min_freq=args.min_freq, top_n=args.top)

        suggestions = []
        if client and result["candidates"]:
            try:
                suggestions = llm_suggest(result, client, model)
                time.sleep(1)
            except Exception as e:
                logger.error(f"LLM suggestion failed for {domain}: {e}")

        suggestions_by_domain[domain] = suggestions
        _print_report(result, suggestions)

    if args.apply and any(suggestions_by_domain.values()):
        if not client:
            print("\n--apply requires LLM_API_KEY to be set")
            sys.exit(1)
        print("\nPatching config.py...")
        _patch_config(suggestions_by_domain)
        print("Done. Review config.py changes before committing.")
    elif not args.apply and any(s for ss in suggestions_by_domain.values() for s in ss):
        print("\n\nRe-run with --apply to patch config.py automatically.")


if __name__ == "__main__":
    main()
