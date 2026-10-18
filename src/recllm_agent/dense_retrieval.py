from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Chunk, SearchResult


class DenseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str = None,
    ) -> None:
        # Resolve model_name to absolute path if it's a relative path
        if model_name is None:
            model_name = "models/bge-small-zh-v1.5"
        if not os.path.isabs(model_name):
            # Assume relative to project root (two levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            candidate = project_root / model_name
            if candidate.exists():
                model_name = str(candidate)
        self.chunks = chunks
        self.embeddings = embeddings
        self.model_name = model_name
        self._norms: np.ndarray | None = None
        self._model: object | None = None

    @property
    def norms(self) -> np.ndarray:
        if self._norms is None:
            self._norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        return self._norms

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vec = self._encode_query(query)
        if query_vec is None:
            return []

        query_vec = query_vec.reshape(1, -1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            return []

        similarities = np.dot(self.embeddings, query_vec.T).flatten()
        similarities = similarities / (self.norms.flatten() * query_norm + 1e-9)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[idx],
                score=float(similarities[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def _encode_query(self, query: str) -> np.ndarray | None:
        try:
            model = self._get_model()
            return model.encode([query], normalize_embeddings=True)[0]
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers"
            )

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(queries, normalize_embeddings=True)

    def search_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        query_vecs = self._encode_queries(queries)
        query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)

        similarities = np.dot(query_vecs, self.embeddings.T)
        chunk_norms = self.norms.flatten()
        similarities = similarities / (query_norms * chunk_norms + 1e-9)

        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]

        results: list[list[SearchResult]] = []
        for qi, idxs in enumerate(top_indices):
            results.append([
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[qi, idx]),
                    rank=rank,
                )
                for rank, idx in enumerate(idxs, start=1)
            ])
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "version": 1,
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1],
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        chunks_path = path.with_suffix(".chunks.jsonl")
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        emb_path = path.with_suffix(".embeddings.npy")
        np.save(emb_path, self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        path = Path(path)

        chunks_path = path.with_suffix(".chunks.jsonl")
        chunks: list[Chunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))

        emb_path = path.with_suffix(".embeddings.npy")
        embeddings = np.load(emb_path)

        state_path = path.with_suffix(".json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            model_name=state.get("model_name", "models/bge-small-zh-v1.5"),
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        model_name: str = "models/bge-small-zh-v1.5",
    ) -> "DenseIndex":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        return cls(chunks=chunks, embeddings=np.array(embeddings), model_name=model_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1],
        }
