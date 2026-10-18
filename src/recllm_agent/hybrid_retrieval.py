from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .dense_retrieval import DenseIndex
from .schema import Chunk, SearchResult


class HybridIndex:
    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        alpha: float = 0.5,
        fusion_method: str = "linear",  # "linear" or "rrf"
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        fusion_method = fusion_method.lower()
        if fusion_method not in {"linear", "rrf"}:
            raise ValueError("fusion_method must be 'linear' or 'rrf'")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.bm25 = bm25_index
        self.dense = dense_index
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k


        self._bm25_min_max: dict[str, float] = {}
        self._dense_min_max: dict[str, float] = {}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidate_k = max(top_k, self.candidate_k)
        bm25_results = self.bm25.search(query, top_k=min(candidate_k, len(self.bm25.chunks)))
        dense_results = self.dense.search(query, top_k=min(candidate_k, len(self.dense.chunks)))

        combined = self._merge_results(bm25_results, dense_results)
        return combined[:top_k]

    def search_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        candidate_k = max(top_k, self.candidate_k)
        bm25_batch = [
            self.bm25.search(q, top_k=min(candidate_k, len(self.bm25.chunks)))
            for q in queries
        ]
        dense_batch = self.dense.search_batch(
            queries, top_k=min(candidate_k, len(self.dense.chunks))
        )

        results: list[list[SearchResult]] = []
        for bm25_results, dense_results in zip(bm25_batch, dense_batch):
            combined = self._merge_results(bm25_results, dense_results)
            results.append(combined[:top_k])
        return results

    def _merge_results(
        self,
        bm25_results: list[SearchResult],
        dense_results: list[SearchResult],
    ) -> list[SearchResult]:
        if not bm25_results and not dense_results:
            return []

        bm25_map = {r.chunk.chunk_id: r.score for r in bm25_results}
        dense_map = {r.chunk.chunk_id: r.score for r in dense_results}
        chunk_map = {
            r.chunk.chunk_id: r.chunk
            for r in bm25_results + dense_results
        }

        all_chunk_ids = set(bm25_map.keys()) | set(dense_map.keys())

        if self.fusion_method == "rrf":
            # Map chunk_id to its rank (1-based index)
            bm25_ranks = {r.chunk.chunk_id: rank for rank, r in enumerate(bm25_results, start=1)}
            dense_ranks = {r.chunk.chunk_id: rank for rank, r in enumerate(dense_results, start=1)}
            
            rrf_scores = {}
            for cid in all_chunk_ids:
                score_bm25 = 1.0 / (self.rrf_k + bm25_ranks[cid]) if cid in bm25_ranks else 0.0
                score_dense = 1.0 / (self.rrf_k + dense_ranks[cid]) if cid in dense_ranks else 0.0
                rrf_scores[cid] = score_bm25 + score_dense
                
            scored: list[tuple[str, float]] = list(rrf_scores.items())
            scored.sort(key=lambda item: (-item[1], item[0]))
        else:
            bm25_scores = np.array([bm25_map.get(cid, 0.0) for cid in all_chunk_ids])
            dense_scores = np.array([dense_map.get(cid, 0.0) for cid in all_chunk_ids])

            bm25_norm = self._minmax_normalize(bm25_scores)
            dense_norm = self._minmax_normalize(dense_scores)

            combined_scores = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

            scored: list[tuple[str, float]] = list(zip(all_chunk_ids, combined_scores))
            scored.sort(key=lambda item: (-item[1], item[0]))

        return [
            SearchResult(
                chunk=chunk_map[cid],
                score=float(score),
                rank=rank,
            )
            for rank, (cid, score) in enumerate(scored, start=1)
        ]

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        score_min = scores.min()
        score_max = scores.max()
        if score_max - score_min < 1e-9:
            return np.ones_like(scores)
        return (scores - score_min) / (score_max - score_min)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "version": 3,
            "alpha": self.alpha,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "bm25_index_path": "indexes/bm25/index.json",
            "dense_index_path": "indexes/dense/index",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        bm25_index: BM25Index | None = None,
        dense_index: DenseIndex | None = None,
        alpha: float = 0.5,
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> "HybridIndex":
        path = Path(path)
        state: dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            alpha = state.get("alpha", alpha)
            if fusion_method is None:
                fusion_method = state.get("fusion_method", "linear")
            if rrf_k is None:
                rrf_k = state.get("rrf_k", 60)
            if candidate_k is None:
                candidate_k = state.get("candidate_k", 50)

        if bm25_index is None:
            bm25_index = BM25Index.load("indexes/bm25/index.json")
        if dense_index is None:
            dense_index = DenseIndex.load("indexes/dense/index")

        return cls(
            bm25_index=bm25_index,
            dense_index=dense_index,
            alpha=alpha,
            fusion_method=fusion_method or "linear",
            rrf_k=60 if rrf_k is None else rrf_k,
            candidate_k=50 if candidate_k is None else candidate_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "alpha": self.alpha,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "num_chunks": len(self.bm25.chunks),
        }


class Reranker:
    def __init__(
        self,
        base_retriever,
        model_name: str = "models/bge-reranker-base",
        candidate_k: int = 20,
    ) -> None:
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self._model: object | None = None

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidates = self.base_retriever.search(query, top_k=self.candidate_k)
        if len(candidates) <= top_k:
            return candidates[:top_k]

        pairs = [(query, c.chunk.text) for c in candidates]
        model = self._get_model()
        scores = model.predict(pairs, show_progress_bar=False)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                chunk=item[0].chunk,
                score=float(item[1]),
                rank=rank,
            )
            for rank, item in enumerate(scored[:top_k], start=1)
        ]

    def search_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        all_candidates = self.base_retriever.search_batch(
            queries, top_k=self.candidate_k
        )

        pairs: list[tuple[str, str]] = []
        pair_counts: list[int] = []
        for query, candidates in zip(queries, all_candidates):
            for c in candidates:
                pairs.append((query, c.chunk.text))
            pair_counts.append(len(candidates))

        model = self._get_model()
        all_scores = model.predict(pairs, show_progress_bar=True)

        results: list[list[SearchResult]] = []
        offset = 0
        for qi, count in enumerate(pair_counts):
            chunk_scores = all_scores[offset : offset + count]
            paired = list(zip(all_candidates[qi], chunk_scores))
            paired.sort(key=lambda x: x[1], reverse=True)

            results.append([
                SearchResult(
                    chunk=item[0].chunk,
                    score=float(item[1]),
                    rank=rank,
                )
                for rank, item in enumerate(paired[:top_k], start=1)
            ])
            offset += count

        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_name": self.model_name,
            "base_retriever": type(self.base_retriever).__name__,
        }
