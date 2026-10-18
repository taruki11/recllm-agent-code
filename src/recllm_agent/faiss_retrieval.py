from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .dense_retrieval import DenseIndex
from .schema import Chunk, SearchResult


class FaissIndex(DenseIndex):

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str = "models/bge-small-zh-v1.5",
        index_type: str = "flat",
        nlist: int = 5,
    ) -> None:
        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required. Install: pip install faiss-cpu")

        super().__init__(chunks, embeddings, model_name)
        self._faiss_index = self._build_faiss_index(embeddings, index_type, nlist)
        self.index_type = index_type
        self._timings: dict[str, float] = {}

    def _build_faiss_index(
        self, embeddings: np.ndarray, index_type: str, nlist: int
    ) -> Any:
        dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings)
            index.add(embeddings)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.add(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        return index

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vec = self._encode_query(query)
        if query_vec is None:
            return []

        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        t0 = time.perf_counter()
        distances, indices = self._faiss_index.search(query_vec, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1:
                break
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(dist),
                rank=rank,
            ))
        self._timings["last_search_ms"] = elapsed_ms
        return results

    def search_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        query_vecs = self._encode_queries(queries).astype(np.float32)

        t0 = time.perf_counter()
        all_distances, all_indices = self._faiss_index.search(query_vecs, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results: list[list[SearchResult]] = []
        for qi, (dists, idxs) in enumerate(zip(all_distances, all_indices)):
            batch_results = []
            for rank, (idx, dist) in enumerate(zip(idxs, dists), start=1):
                if idx == -1:
                    break
                batch_results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(dist),
                    rank=rank,
                ))
            results.append(batch_results)
        self._timings["last_batch_ms"] = elapsed_ms
        return results

    @property
    def timing_summary(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type,
            "num_vectors": self._faiss_index.ntotal,
            "dim": self._faiss_index.d,
            "last_search_ms": self._timings.get("last_search_ms", 0),
            "last_batch_ms": self._timings.get("last_batch_ms", 0),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        super().save(path)
        faiss.write_index(self._faiss_index, str(path.with_suffix(".faiss")))

    @classmethod
    def load(cls, path: str | Path, index_type: str = "flat") -> "FaissIndex":
        path = Path(path)
        dense = DenseIndex.load(path)

        faiss_path = path.with_suffix(".faiss")
        if not faiss_path.exists():
            return cls.from_dense(dense, index_type=index_type)

        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required")

        faiss_index = faiss.read_index(str(faiss_path))
        instance = cls(
            chunks=dense.chunks,
            embeddings=dense.embeddings,
            model_name=dense.model_name,
            index_type=index_type,
        )
        instance._faiss_index = faiss_index
        return instance

    @classmethod
    def from_dense(cls, dense: DenseIndex, index_type: str = "flat") -> "FaissIndex":
        return cls(
            chunks=dense.chunks,
            embeddings=dense.embeddings,
            model_name=dense.model_name,
            index_type=index_type,
        )

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["faiss_index_type"] = self.index_type
        base["faiss_ntotal"] = self._faiss_index.ntotal
        return base
