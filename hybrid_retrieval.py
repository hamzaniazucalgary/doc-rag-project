"""
Hybrid retrieval combining BM25 (keyword) + semantic search with cross-encoder reranking.

Pipeline:
1. Retrieve top-N candidates from semantic search
2. Retrieve top-N candidates from BM25
3. Merge using Reciprocal Rank Fusion (RRF)
4. Rerank merged candidates with cross-encoder
5. Return top-K results
"""
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    TOP_K,
    TOP_K_CANDIDATES,
    HYBRID_ALPHA,
    RERANK_MODEL,
    SIMILARITY_THRESHOLD
)
from storage import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Standardized retrieval result."""
    content: str
    doc_name: str
    doc_id: str
    page: int
    chunk_index: int
    score: float  # Higher is better (reranker score or fused score)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "doc_name": self.doc_name,
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": self.score
        }


class HybridRetriever:
    """
    Hybrid retriever combining BM25 + semantic search with cross-encoder reranking.
    
    This significantly improves retrieval quality by:
    1. BM25 catches exact keyword matches that semantic search misses
    2. Semantic search catches meaning even with different words
    3. Cross-encoder provides fine-grained relevance scoring
    """
    
    def __init__(
        self,
        store: VectorStore,
        rerank_model: str = RERANK_MODEL,
        enable_reranking: bool = True
    ):
        self.store = store
        self.enable_reranking = enable_reranking
        
        # BM25 index (built lazily)
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: list[dict] = []
        self.chunk_id_to_idx: dict[str, int] = {}
        
        # Cross-encoder reranker
        if enable_reranking:
            logger.info(f"Loading reranker model: {rerank_model}")
            self.reranker = CrossEncoder(rerank_model, max_length=512)
        else:
            self.reranker = None
    
    def build_bm25_index(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dicts with "id", "content", and "metadata" keys
        """
        logger.info(f"Building BM25 index with {len(chunks)} chunks")
        
        self.corpus_chunks = chunks
        self.chunk_id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
        
        # Tokenize for BM25 (simple whitespace tokenization)
        tokenized_corpus = [
            self._tokenize(c["content"]) for c in chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully")
    
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        n_results: int = TOP_K,
        n_candidates: int = TOP_K_CANDIDATES,
        alpha: float = HYBRID_ALPHA,
        use_reranking: bool = True
    ) -> tuple[list[RetrievalResult], bool]:
        """
        Hybrid retrieval pipeline.
        
        Args:
            query: User query string
            query_embedding: Embedding vector for query
            n_results: Number of final results to return
            n_candidates: Number of candidates to retrieve before reranking
            alpha: Weight for semantic vs BM25 (1.0 = pure semantic, 0.0 = pure BM25)
            use_reranking: Whether to apply cross-encoder reranking
        
        Returns:
            Tuple of (results, is_low_confidence)
        """
        # Step 1: Semantic retrieval
        semantic_results = self.store.query(query_embedding, n_results=n_candidates)
        
        # Step 2: BM25 retrieval (if index exists)
        bm25_results = []
        if self.bm25 is not None:
            bm25_results = self._bm25_retrieve(query, n_candidates)
        
        # Step 3: Merge candidates
        merged = self._merge_candidates(semantic_results, bm25_results, alpha)
        
        # Step 4: Rerank with cross-encoder
        if use_reranking and self.reranker is not None and merged:
            merged = self._rerank(query, merged)
        
        # Step 5: Take top-K
        final_results = merged[:n_results]
        
        # Determine confidence
        is_low_confidence = self._check_low_confidence(final_results)
        
        return final_results, is_low_confidence
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _bm25_retrieve(self, query: str, n_results: int) -> list[dict]:
        """Retrieve using BM25."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-N indices
        top_indices = np.argsort(scores)[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                chunk = self.corpus_chunks[idx]
                results.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "bm25_score": float(scores[idx])
                })
        
        return results
    
    def _merge_candidates(
        self,
        semantic_results: dict,
        bm25_results: list[dict],
        alpha: float
    ) -> list[RetrievalResult]:
        """
        Merge semantic and BM25 results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) across all rankings
        where k=60 is standard constant
        """
        k = 60  # RRF constant
        scores = {}  # id -> {"rrf_score": float, "data": dict}
        
        # Score semantic results
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(
            semantic_results.get("ids", []),
            semantic_results.get("documents", []),
            semantic_results.get("metadatas", []),
            semantic_results.get("distances", [])
        )):
            rrf_score = alpha * (1 / (k + rank + 1))
            scores[doc_id] = {
                "rrf_score": rrf_score,
                "content": doc,
                "metadata": meta or {},
                "semantic_dist": dist
            }
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_contribution = (1 - alpha) * (1 / (k + rank + 1))
            
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += rrf_contribution
            else:
                scores[doc_id] = {
                    "rrf_score": rrf_contribution,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_dist": 1.0  # No semantic score
                }
        
        # Convert to RetrievalResult and sort by RRF score
        results = []
        for doc_id, data in scores.items():
            meta = data["metadata"]
            results.append(RetrievalResult(
                content=data["content"],
                doc_name=meta.get("doc_name", "Unknown"),
                doc_id=meta.get("doc_id", ""),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", 0),
                score=data["rrf_score"]
            ))
        
        # Sort by RRF score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [(query, c.content) for c in candidates]
        
        # Get cross-encoder scores
        ce_scores = self.reranker.predict(pairs)
        
        # Update scores and re-sort
        for i, candidate in enumerate(candidates):
            candidate.score = float(ce_scores[i])
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates
    
    def _check_low_confidence(self, results: list[RetrievalResult]) -> bool:
        """Check if results are low confidence."""
        if not results:
            return True
        
        # If reranking is enabled, use reranker scores
        # Cross-encoder scores are typically in [-10, 10] range
        # Scores < 0 generally indicate poor relevance
        if self.reranker is not None:
            top_score = results[0].score
            return top_score < 0
        
        # Fallback: no good way to determine without reranker
        return False


def create_retrieval_pipeline(store: VectorStore, enable_reranking: bool = True):
    """
    Factory function to create hybrid retriever with BM25 index.
    
    Should be called after documents are ingested to build the BM25 index.
    """
    retriever = HybridRetriever(store, enable_reranking=enable_reranking)
    
    # Build BM25 index from existing chunks in store
    all_data = store.collection.get(include=["documents", "metadatas"])
    
    if all_data["ids"]:
        chunks = []
        for i, doc_id in enumerate(all_data["ids"]):
            chunks.append({
                "id": doc_id,
                "content": all_data["documents"][i],
                "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {}
            })
        
        retriever.build_bm25_index(chunks)
    
    return retriever
