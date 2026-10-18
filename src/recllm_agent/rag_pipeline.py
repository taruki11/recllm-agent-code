from __future__ import annotations

from .bm25 import BM25Index
from .schema import SearchResult


_CONTEXT_METADATA_PATTERNS = (
    "score=",
    "chunk=",
    "chunk_id",
    "doc_id",
    "source_path",
    "prompt_template",
    "retrieved_chunk_ids",
    "retrieved_doc_ids",
)


def _clean_context_text(text: str) -> str:
    text = text.replace("\r", "\n")
    kept_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(pattern in lowered for pattern in _CONTEXT_METADATA_PATTERNS):
            continue
        kept_lines.append(line)
    return " ".join(kept_lines).strip()


def _truncate_without_metadata(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()
    sentence_ends = [clipped.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?")]
    boundary = max(sentence_ends)
    if boundary >= max_chars // 2:
        return clipped[: boundary + 1].strip()
    return clipped.strip()


def format_context(results: list[SearchResult], max_chars_per_chunk: int = 500) -> str:
    blocks = []
    for result in results:
        title = result.chunk.title.strip()
        text = _clean_context_text(result.chunk.text)
        text = _truncate_without_metadata(text, max_chars_per_chunk)
        blocks.append(
            f"[{result.rank}] {title}\n{text}"
        )
    return "\n\n".join(blocks)


def make_extractive_answer(question: str, results: list[SearchResult]) -> str:
    """Generate a transparent baseline answer without calling an LLM."""

    if not results:
        return (
            "没有在当前知识库中检索到足够相关的材料。"
            "建议先补充论文、项目文档或 JD 到 data/raw_docs。"
        )

    citations = ", ".join(f"[{result.rank}]" for result in results[:3])
    lines = [
        f"问题：{question}",
        "",
        "Baseline 回答：",
        f"我先基于检索到的资料回答，主要依据 {citations}。这不是最终 LLM 生成结果，而是可运行的 RAG baseline。",
        "",
        "关键依据：",
    ]
    for result in results[:3]:
        snippet = result.chunk.text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        lines.append(f"- [{result.rank}] {result.chunk.title}: {snippet}")

    lines.extend(
        [
            "",
            "后续升级点：接入 Qwen 生成器后，应把上面的 top-k chunk 作为 context，要求模型给出带引用、不过度声称的答案。",
        ]
    )
    return "\n".join(lines)


class BaselineRAG:
    def __init__(self, index: BM25Index) -> None:
        self.index = index

    @classmethod
    def from_index_path(cls, path: str) -> "BaselineRAG":
        return cls(BM25Index.load(path))

    def answer(self, question: str, top_k: int = 5) -> tuple[str, list[SearchResult]]:
        results = self.index.search(question, top_k=top_k)
        return make_extractive_answer(question, results), results
