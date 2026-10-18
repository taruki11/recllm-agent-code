from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    doc_type: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            source_path=str(data["source_path"]),
            text=str(data["text"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

    def citation(self) -> str:
        return f"[{self.rank}] {self.chunk.title}::{self.chunk.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk": self.chunk.to_dict(),
            "citation": self.citation(),
        }
