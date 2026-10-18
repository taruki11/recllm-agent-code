from __future__ import annotations

import re
from typing import Any
from .schema import Chunk, SearchResult

class SentenceWindowRetriever:
    """Advanced Sentence Window Retriever that expands matching sentences with adjacent context."""
    
    def __init__(self, base_retriever, window_size: int = 1) -> None:
        self.base_retriever = base_retriever
        self.window_size = window_size

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        """Splits a paragraph into sentences, preserving LaTeX formulas."""
        # Simple splitting by sentence-ending punctuation but avoiding splitting within inline math $...$
        # Splits by 。|！|？ or newlines
        pattern = re.compile(r'([^。！？\n]+(?:。|！|？|\n|$))')
        sentences = pattern.findall(text)
        return [s.strip() for s in sentences if s.strip()]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # 1. Retrieve candidates from base retriever (e.g. HybridIndex or Reranker)
        candidates = self.base_retriever.search(query, top_k=top_k)
        
        # 2. Apply sentence window expansion on each candidate chunk
        expanded_candidates = []
        for c in candidates:
            chunk = c.chunk
            text = chunk.text
            
            # Split paragraph into sentences
            sentences = self.split_into_sentences(text)
            if not sentences:
                expanded_candidates.append(c)
                continue
                
            # Find the most query-relevant sentence in the chunk based on keyword overlap
            best_idx = 0
            best_overlap = -1
            query_words = set(list(query.lower()))  # Character/word set overlap for Chinese/English
            
            for idx, sent in enumerate(sentences):
                overlap = len(set(list(sent.lower())) & query_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            
            # Expand window: [best_idx - window_size, best_idx + window_size]
            start_idx = max(0, best_idx - self.window_size)
            end_idx = min(len(sentences) - 1, best_idx + self.window_size)
            
            window_text = " ".join(sentences[start_idx : end_idx + 1])
            
            # Create a new expanded chunk
            expanded_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_path=chunk.source_path,
                text=window_text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata
            )
            
            expanded_candidates.append(SearchResult(
                chunk=expanded_chunk,
                score=c.score,
                rank=c.rank
            ))
            
        return expanded_candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "window_size": self.window_size,
            "base_retriever": type(self.base_retriever).__name__
        }
