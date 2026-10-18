from __future__ import annotations

import hashlib
from pathlib import Path

from .schema import Document

SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".jsonl"}


def stable_doc_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    stem = path.stem.lower().replace(" ", "_")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def load_text_documents(raw_dir: str | Path) -> list[Document]:
    """Load Markdown/text-like files as Documents.

    PDF parsing is deliberately left for a later dependency-enabled stage. The
    baseline starts with Markdown/text so every agent can run it immediately.
    """

    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw doc dir not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        documents.append(
            Document(
                doc_id=stable_doc_id(path, root),
                title=path.stem,
                source_path=str(path),
                doc_type=path.suffix.lower().lstrip("."),
                text=text,
                metadata={"relative_path": path.relative_to(root).as_posix()},
            )
        )
    return documents
