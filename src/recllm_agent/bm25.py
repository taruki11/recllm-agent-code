from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import Chunk, SearchResult
from .tokenization import tokenize


class BM25Index:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenized_chunks: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.tokenized_chunks = tokenized_chunks
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(tokens) for tokens in tokenized_chunks]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in tokenized_chunks]
        self.idf = self._build_idf()

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        return cls(chunks, [tokenize(chunk.text) for chunk in chunks], k1=k1, b=b)

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.tokenized_chunks)
        df: Counter[str] = Counter()
        for tokens in self.tokenized_chunks:
            df.update(set(tokens))
        return {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        scored: list[tuple[int, float]] = []
        for idx, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[idx] or 1
            for term, qf in query_counts.items():
                tf = term_freq.get(term, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += qf * idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=self.chunks[idx], score=score, rank=rank)
            for rank, (idx, score) in enumerate(scored[:top_k], start=1)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "tokenized_chunks": self.tokenized_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Index":
        chunks = [Chunk.from_dict(item) for item in data["chunks"]]
        return cls(
            chunks=chunks,
            tokenized_chunks=[list(tokens) for tokens in data["tokenized_chunks"]],
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )

    def save(self, path: str | Path) -> None:
        from .io_utils import write_json

        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        from .io_utils import read_json

        return cls.from_dict(read_json(path))  # type: ignore[arg-type]
