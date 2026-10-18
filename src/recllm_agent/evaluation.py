from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from .schema import SearchResult


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ...


def load_eval_questions(path: str | Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def canonical_doc_id(value: str) -> str:
    """Normalize document ids so stable hash suffixes and titles can be compared."""
    normalized = str(value).strip().lower().replace("\\", "/").rsplit("/", 1)[-1]
    normalized = re.sub(r"\.(md|markdown|txt|json|jsonl)$", "", normalized)
    return re.sub(r"_[0-9a-f]{8}$", "", normalized)


def is_relevant(result_doc_id: str, result_chunk_id: str, question: dict[str, Any]) -> bool:
    relevant_doc_ids = set(question.get("relevant_doc_ids", []))
    relevant_chunk_ids = set(question.get("relevant_chunk_ids", []))
    canonical_relevant_doc_ids = {canonical_doc_id(value) for value in relevant_doc_ids}
    return (
        canonical_doc_id(result_doc_id) in canonical_relevant_doc_ids
        or result_chunk_id in relevant_chunk_ids
    )


def evaluate_retrieval(index: Retriever, questions: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    hits = 0
    recalls = []
    reciprocal_ranks = []
    details = []

    if hasattr(index, "search_batch"):
        queries = [q["question"] for q in questions]
        all_results = index.search_batch(queries, top_k=top_k)
        pairs = list(zip(questions, all_results))
    else:
        pairs = [(q, index.search(q["question"], top_k=top_k)) for q in questions]

    for question, results in pairs:
        first_rank = None
        relevant_doc_ids = {
            canonical_doc_id(value)
            for value in question.get("relevant_doc_ids", [])
        }
        relevant_chunk_ids = set(question.get("relevant_chunk_ids", []))
        matched_relevant = set()
        for result in results:
            if is_relevant(result.chunk.doc_id, result.chunk.chunk_id, question):
                canonical_result_doc_id = canonical_doc_id(result.chunk.doc_id)
                if canonical_result_doc_id in relevant_doc_ids:
                    matched_relevant.add(("doc", canonical_result_doc_id))
                if result.chunk.chunk_id in relevant_chunk_ids:
                    matched_relevant.add(("chunk", result.chunk.chunk_id))
                if first_rank is None:
                    first_rank = result.rank
        hit = first_rank is not None
        hits += int(hit)
        relevant_count = len(relevant_doc_ids) + len(relevant_chunk_ids)
        recalls.append(len(matched_relevant) / relevant_count if relevant_count else 0.0)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        details.append(
            {
                "id": question.get("id"),
                "question": question["question"],
                "hit": hit,
                "recall": recalls[-1],
                "first_rank": first_rank,
                "top_results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "doc_id": result.chunk.doc_id,
                        "chunk_id": result.chunk.chunk_id,
                        "title": result.chunk.title,
                    }
                    for result in results
                ],
            }
        )

    total = max(len(questions), 1)
    return {
        "top_k": top_k,
        "num_questions": len(questions),
        "hit_rate_at_k": hits / total,
        "recall_at_k": sum(recalls) / total,
        "mrr": sum(reciprocal_ranks) / total,
        "details": details,
    }


def evaluate_answer_quality(
    questions: list[dict[str, Any]],
    all_results: list[list[SearchResult]],
    answers: list[str],
    cited_count: int = 3,
) -> dict[str, Any]:
    total_cited = 0
    total_cited_relevant = 0
    total_relevant_cited = 0
    total_relevant_available = 0
    grounded_count = 0
    details = []

    for question, results, answer in zip(questions, all_results, answers):
        citation_numbers = []
        for raw_number in re.findall(r"\[(\d+)\]", answer):
            number = int(raw_number)
            if number not in citation_numbers:
                citation_numbers.append(number)
        if cited_count > 0:
            citation_numbers = citation_numbers[:cited_count]

        cited = [
            results[number - 1]
            for number in citation_numbers
            if 1 <= number <= len(results)
        ]
        invalid_citations = [
            number for number in citation_numbers
            if number < 1 or number > len(results)
        ]
        cited_chunk_ids = [r.chunk.chunk_id for r in cited]
        cited_doc_ids = [r.chunk.doc_id for r in cited]

        relevant_mask = [
            is_relevant(r.chunk.doc_id, r.chunk.chunk_id, question)
            for r in cited
        ]
        cited_relevant = sum(relevant_mask)

        all_relevant = [
            is_relevant(r.chunk.doc_id, r.chunk.chunk_id, question)
            for r in results
        ]
        relevant_available = sum(all_relevant)

        total_cited += len(citation_numbers)
        total_cited_relevant += cited_relevant
        total_relevant_cited += cited_relevant
        total_relevant_available += relevant_available
        if cited_relevant > 0:
            grounded_count += 1

        details.append({
            "id": question.get("id"),
            "question": question["question"],
            "cited_chunks": cited_chunk_ids,
            "cited_doc_ids": cited_doc_ids,
            "citation_numbers": citation_numbers,
            "invalid_citations": invalid_citations,
            "cited_relevant_count": cited_relevant,
            "total_cited": len(citation_numbers),
            "relevant_in_topk": relevant_available,
            "grounded": cited_relevant > 0,
            "answer_snippet": answer[:300],
        })

    total_q = max(len(questions), 1)
    return {
        "num_questions": len(questions),
        "cited_per_question": cited_count,
        "citation_precision": total_cited_relevant / max(total_cited, 1),
        "citation_recall": total_relevant_cited / max(total_relevant_available, 1),
        "groundedness": grounded_count / total_q,
        "total_cited": total_cited,
        "total_cited_relevant": total_cited_relevant,
        "total_relevant_available": total_relevant_available,
        "details": details,
    }
