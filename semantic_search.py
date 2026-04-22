"""
Semantic Search Module (Upgrade)
Uses local BGE embedding model for intelligent search
"""

import os
import json
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


# BGE-M3: multilingual model, works well for English academic papers
DEFAULT_MODEL = 'BAAI/bge-m3'


class SemanticSearch:
    """Semantic search using local embedding model (free)"""

    def __init__(self, papers_path: str = "output/papers.json", model_name: str = DEFAULT_MODEL):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        self.papers_path = papers_path
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        print(f"Loading papers from {papers_path}...")
        with open(papers_path, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)

        self.embeddings = self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load cached embeddings or compute new ones"""
        cache_path = "output/embeddings.npy"
        idx_cache = "output/embeddings_idx.json"

        # Check if cached embeddings exist and match papers
        if os.path.exists(cache_path) and os.path.exists(idx_cache):
            cached_idx = json.load(open(idx_cache, 'r'))
            current_ids = [p.get('id', p.get('arxiv_url', '')) for p in self.papers]

            if cached_idx == current_ids:
                print("Loading cached embeddings...")
                return np.load(cache_path)

        print("Computing embeddings (this may take a few minutes on first run)...")
        texts = [self._paper_to_text(p) for p in self.papers]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        # Cache embeddings
        np.save(cache_path, embeddings)
        current_ids = [p.get('id', p.get('arxiv_url', '')) for p in self.papers]
        with open(idx_cache, 'w') as f:
            json.dump(current_ids, f)

        print(f"Embeddings cached to {cache_path}")
        return embeddings

    def _paper_to_text(self, paper: dict) -> str:
        """Convert paper to text for embedding"""
        parts = [
            paper.get('title', ''),
            paper.get('abstract', ''),
            ' '.join(paper.get('_domains', [])),
            ' '.join(paper.get('_tasks', []))
        ]
        return ' '.join(parts)

    def search(self, query: str, top_k: int = 10, domain_filter: str = None) -> list:
        """Semantic search"""
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding[0])

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]  # Get more for filtering

        results = []
        for idx in top_indices:
            paper = dict(self.papers[idx])

            # Filter by domain if specified
            if domain_filter and domain_filter not in paper.get('_domains', []):
                continue

            paper['_score'] = float(similarities[idx])
            paper['_rank'] = len(results) + 1

            # Remove large fields to reduce response size
            if 'abstract' in paper and len(paper['abstract']) > 500:
                paper['abstract'] = paper['abstract'][:500] + '...'

            results.append(paper)

            if len(results) >= top_k:
                break

        return results

    def search_by_domain(self, query: str, domain: str, top_k: int = 10) -> list:
        """Search within a specific domain"""
        return self.search(query, top_k, domain_filter=domain)

    def find_similar(self, paper_id: str, top_k: int = 5) -> list:
        """Find similar papers to a given paper"""
        # Find paper index
        idx = None
        for i, p in enumerate(self.papers):
            if p.get('id') == paper_id or p.get('arxiv_url', '').endswith(paper_id):
                idx = i
                break

        if idx is None:
            return []

        paper_embedding = self.embeddings[idx:idx+1]
        similarities = np.dot(self.embeddings, paper_embedding[0])

        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip self

        return [dict(self.papers[i]) for i in top_indices]


def main():
    """Demo usage"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Please install sentence-transformers first:")
        print("  pip install sentence-transformers")
        return

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--domain", choices=["world_model", "physical_ai", "medical_ai"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    search = SemanticSearch()
    results = search.search(args.query, args.top_k, args.domain)

    print(f"\nFound {len(results)} results for: {args.query}\n")

    for paper in results:
        print(f"[{paper['_rank']}] Score: {paper['_score']:.3f}")
        print(f"Title: {paper['title']}")
        print(f"Domains: {paper.get('_domains', [])}")
        print(f"Tasks: {paper.get('_tasks', [])}")
        print(f"URL: {paper.get('arxiv_url', '')}")
        print("-" * 50)


if __name__ == "__main__":
    main()