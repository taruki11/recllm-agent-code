from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recllm_agent.bm25 import BM25Index
from recllm_agent.dense_retrieval import DenseIndex
from recllm_agent.hybrid_retrieval import HybridIndex
from recllm_agent.evaluation import load_eval_questions
from recllm_agent.qwen_gen import build_citation_prompt
from recllm_agent.rag_pipeline import format_context
from recllm_agent.schema import SearchResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/bm25/index.json")
    parser.add_argument("--dense-index", default="indexes/dense/index")
    parser.add_argument("--eval", required=True, help="Question source JSONL; evaluation sources are rejected by default")
    parser.add_argument("--allow-eval-source", action="store_true")
    parser.add_argument("--out", default="data/sft/instruction.jsonl")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--fusion-method", choices=["linear", "rrf"], default="rrf")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def validate_training_source(path: str, allow_eval_source: bool = False) -> None:
    parts = {part.lower() for part in Path(path).parts}
    if "eval" in parts and not allow_eval_source:
        raise ValueError(
            f"Refusing to build training data from evaluation source: {path}. "
            "Use a disjoint training source or pass --allow-eval-source explicitly."
        )


def build_sft_sample(
    question: str,
    results: list[SearchResult],
    system: str = "",
) -> dict:
    context = format_context(results)
    prompt = build_citation_prompt(question, results)

    return {
        "instruction": question,
        "input": context,
        "output": "",
        "system": system,
        "prompt_template": prompt,
        "retrieved_chunk_ids": [r.chunk.chunk_id for r in results],
        "retrieved_doc_ids": [r.chunk.doc_id for r in results],
    }


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

    samples = []
    for question, results in zip(questions, all_results):
        sample = build_sft_sample(question["question"], results)
        sample["eval_id"] = question.get("id", "")
        sample["relevant_doc_ids"] = question.get("relevant_doc_ids", [])
        samples.append(sample)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"sft_samples={len(samples)}")
    print(f"out={args.out}")
    print("next: fill 'output' field with Qwen-generated answers or manual review")


if __name__ == "__main__":
    main()
