from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Chunk


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data: object) -> None:
    ensure_parent(path)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_chunks_jsonl(path: str | Path, chunks: Iterable[Chunk]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8", newline="\n") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
