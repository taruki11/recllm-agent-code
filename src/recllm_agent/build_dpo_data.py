from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recllm_agent.evaluation import load_eval_questions, is_relevant
from recllm_agent.build_sft_data import validate_training_source
from recllm_agent.bm25 import BM25Index
from recllm_agent.dense_retrieval import DenseIndex
from recllm_agent.hybrid_retrieval import HybridIndex
from recllm_agent.rag_pipeline import format_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/bm25/index.json")
    parser.add_argument("--dense-index", default="indexes/dense/index")
    parser.add_argument("--eval", required=True, help="Question source JSONL; evaluation sources are rejected by default")
    parser.add_argument("--allow-eval-source", action="store_true")
    parser.add_argument("--out", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--fusion-method", choices=["linear", "rrf"], default="rrf")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def build_dpo_pair(
    question: dict,
    results: list,
    retrieved_doc_ids: list[str],
) -> dict | None:
    relevant_mask = [
        is_relevance(r.chunk.doc_id, r.chunk.chunk_id, question)
        for r in results
    ]
    relevant_results = [r for r, m in zip(results, relevant_mask) if m]
    irrelevant_results = [r for r, m in zip(results, relevant_mask) if not m]

    if not relevant_results:
        return None

    chosen_context = format_context(relevant_results)
    rejected_context = format_context(irrelevant_results) if irrelevant_results else format_context(results[:1])
    common_context = format_context(results)

    prompt = f"""你是一个推荐系统领域的学习助手。请根据以下资料回答问题。

资料：
{common_context}

问题：{question['question']}

请用简洁中文回答，标注引用来源如 [1]、[2]。"""

    return {
        "id": question.get("id", ""),
        "question": question["question"],
        "prompt": prompt,
        "chosen": "",
        "rejected": "",
        "status": "requires_preference_answers",
        "positive_evidence": chosen_context,
        "negative_evidence": rejected_context,
        "relevant_doc_ids": question.get("relevant_doc_ids", []),
        "chosen_chunk_ids": [r.chunk.chunk_id for r in relevant_results],
        "rejected_chunk_ids": [r.chunk.chunk_id for r in irrelevant_results],
    }


def is_relevance(doc_id: str, chunk_id: str, question: dict) -> bool:
    return is_relevant(doc_id, chunk_id, question)


def main() -> None:
    args = parse_args()
    validate_training_source(args.eval, args.allow_eval_source)

    bm25 = BM25Index.load(args.index)
    dense = DenseIndex.load(args.dense_index)
    retriever = HybridIndex(
        bm25_index=bm25,
        dense_index=dense,
        alpha=args.alpha,
        fusion_method=args.fusion_method,
        rrf_k=args.rrf_k,
    )

    questions = load_eval_questions(args.eval)
    queries = [q["question"] for q in questions]
    all_results = retriever.search_batch(queries, top_k=args.top_k)

    pairs = []
    for question, results in zip(questions, all_results):
        pair = build_dpo_pair(question, results, None)
        if pair:
            pairs.append(pair)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"dpo_pairs={len(pairs)}")
    print(f"out={args.out}")
    print("next: fill 'chosen' and 'rejected' with preference answers generated from the same prompt")


if __name__ == "__main__":
    main()
