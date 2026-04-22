"""
LLM-based paper classification using any OpenAI-compatible API.
- Replaces regex domain matching with semantic understanding
- Discovers dynamic subtopic tags instead of hardcoded task labels
- Caches results so papers are only classified once

Supported providers (set env vars):
  智谱 GLM-4-Flash (free, recommended):
    LLM_API_KEY=<your_key>
    LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    LLM_MODEL=glm-4-flash

  DeepSeek (cheap):
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_MODEL=deepseek-chat

  Gemini (free tier, OpenAI-compatible endpoint):
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    LLM_MODEL=gemini-2.0-flash

  Groq (free tier):
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=llama-3.3-70b-versatile
"""

import json
import logging
import os
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

CACHE_FILE = "output/llm_classify_cache.json"
BATCH_SIZE = 10  # papers per API call

# Defaults target 智谱 GLM-4-Flash (completely free)
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"

DOMAIN_DESCRIPTIONS = {
    "world_model": (
        "World Models: video generation/prediction, neural radiance fields (NeRF), "
        "3D Gaussian splatting, model-based reinforcement learning, sim-to-real transfer, "
        "embodied agents with world models, scene representation learning, latent dynamics"
    ),
    "physical_ai": (
        "Physical AI: physics-informed neural networks (PINN), neural operators (FNO/DeepONet), "
        "robotics/manipulation/grasping, embodied intelligence, fluid dynamics, climate modeling, "
        "molecular dynamics, material simulation, physical reasoning"
    ),
    "medical_ai": (
        "Medical AI: medical image analysis (MRI/CT/X-ray), pathology/histopathology, "
        "cancer detection/segmentation, drug discovery/molecular design, protein structure, "
        "clinical decision support, surgical robotics, medical VLMs/LLMs, health monitoring"
    ),
}


def _load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict):
    os.makedirs("output", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _build_prompt(papers: List[Dict]) -> str:
    papers_text = "\n\n".join(
        f"[{i+1}] Title: {p.get('title', '')}\nAbstract: {p.get('abstract', '')[:400]}"
        for i, p in enumerate(papers)
    )
    domains_desc = "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DESCRIPTIONS.items())

    return f"""Classify each research paper and extract its specific research subtopics.

Domain definitions:
{domains_desc}

Papers to classify:
{papers_text}

Return a JSON array with exactly {len(papers)} objects, one per paper in order:
[
  {{
    "index": 1,
    "domains": ["world_model"],
    "tags": ["video diffusion model", "future frame prediction"],
    "type": "Method"
  }}
]

Rules:
- domains: subset of [world_model, physical_ai, medical_ai]. Empty list if not relevant to any.
- tags: 2-4 specific research subtopics as lowercase strings (e.g. "3d gaussian splatting", "robotic grasping"). Be specific, not generic.
- type: "Method", "Dataset", or "Survey" only.

Return only valid JSON, no explanation."""


def _classify_batch(papers: List[Dict], client, model: str) -> List[Dict]:
    """Call LLM to classify a batch of papers. Returns list of classification dicts."""
    prompt = _build_prompt(papers)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.split("```")[0].strip()

    return json.loads(text)


def classify_papers_with_llm(papers: List[Dict]) -> List[Dict]:
    """
    Classify papers via any OpenAI-compatible API with persistent caching.
    Falls back gracefully if LLM_API_KEY is not set.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — skipping LLM classification")
        return papers

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("LLM_API_KEY not set — skipping LLM classification")
        return papers

    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    logger.info(f"LLM provider: base_url={base_url}, model={model}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = _load_cache()

    uncached = [(i, p) for i, p in enumerate(papers) if p.get("id") not in cache]
    logger.info(f"LLM classifying {len(uncached)} papers ({len(papers) - len(uncached)} from cache)")

    save_interval = 5
    for batch_num, batch_start in enumerate(range(0, len(uncached), BATCH_SIZE)):
        batch = uncached[batch_start:batch_start + BATCH_SIZE]
        indices = [x[0] for x in batch]
        batch_papers = [x[1] for x in batch]

        try:
            results = _classify_batch(batch_papers, client, model)

            for result in results:
                idx_in_batch = result.get("index", 0) - 1
                if not (0 <= idx_in_batch < len(batch_papers)):
                    continue
                paper_idx = indices[idx_in_batch]
                paper_id = batch_papers[idx_in_batch].get("id", "")

                classification = {
                    "domains": result.get("domains", []),
                    "tags": [t.lower().strip() for t in result.get("tags", [])],
                    "type": result.get("type", "Method"),
                }
                cache[paper_id] = classification

                papers[paper_idx]["_domains"] = classification["domains"]
                papers[paper_idx]["_tags"] = classification["tags"]
                papers[paper_idx]["type"] = classification["type"]

        except Exception as e:
            logger.error(f"LLM batch {batch_num} failed: {e} — keeping regex results")

        if batch_num % save_interval == 0:
            _save_cache(cache)

        time.sleep(0.3)

    _save_cache(cache)

    # Apply cached results to papers that were already cached
    for paper in papers:
        paper_id = paper.get("id", "")
        if paper_id in cache and "_tags" not in paper:
            cached = cache[paper_id]
            paper["_domains"] = cached.get("domains", paper.get("_domains", []))
            paper["_tags"] = cached.get("tags", [])
            if not paper.get("type"):
                paper["type"] = cached.get("type", "Method")

    logger.info("LLM classification complete")
    return papers
