from __future__ import annotations
from typing import Any, Dict, List

class Memory:
    """Session Memory to store user profile, query history, and retrieved context cache.
    Aligns with DeepLearning.AI Pattern 3: Planning & Memory."""
    def __init__(self) -> None:
        self.user_profile: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.context_cache: List[Any] = []

    def set_user_profile(self, key: str, value: Any) -> None:
        self.user_profile[key] = value

    def get_user_profile(self, key: str, default: Any = None) -> Any:
        return self.user_profile.get(key, default)

    def add_history(self, question: str, answer: str) -> None:
        self.history.append({
            "question": str(question),
            "answer": str(answer)
        })
        # Keep only the last 5 turns to prevent model context explosion
        if len(self.history) > 5:
            self.history.pop(0)

    def get_history_summary(self) -> str:
        if not self.history:
            return "无历史问答记录。"
        summary = ""
        for idx, turn in enumerate(self.history, 1):
            summary += f"历史第 {idx} 轮问答：\n提问：{turn['question']}\n回答：{turn['answer']}\n\n"
        return summary

    def cache_context(self, context_list: List[Any]) -> None:
        self.context_cache.extend(context_list)
        # Keep only the last 20 cached chunks
        if len(self.context_cache) > 20:
            self.context_cache = self.context_cache[-20:]

    def get_cached_context_str(self, limit: int = 5) -> str:
        if not self.context_cache:
            return ""
        context_str = ""
        # Deduplicate
        seen = set()
        deduped = []
        for r in self.context_cache:
            key = (r.chunk.title, r.chunk.text)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                
        # Interleave retrieval batches by rank so one early tool call cannot
        # consume the entire evidence budget in a multi-step query.
        selected = sorted(deduped, key=lambda result: result.rank)[:limit]
        for idx, r in enumerate(selected, 1):
            context_str += f"[{idx}] {r.chunk.title}: {r.chunk.text[:400]}\n\n"
        return context_str

    def clear_context(self) -> None:
        self.context_cache.clear()

    def clear(self) -> None:
        self.user_profile.clear()
        self.history.clear()
        self.clear_context()
