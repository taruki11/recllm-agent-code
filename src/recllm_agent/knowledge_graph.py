from __future__ import annotations

import re
from typing import Any

class RecommendationKnowledgeGraph:
    """Lightweight Local Knowledge Graph representing Recommendation System concepts for multi-hop expansion."""
    
    def __init__(self) -> None:
        # Define entity nodes: Concept Name -> metadata
        self.nodes: dict[str, dict[str, Any]] = {
            "SASRec": {"type": "Model", "desc": "Self-Attention Based Sequential Recommendation"},
            "GRU4Rec": {"type": "Model", "desc": "Sequential Recommendation using RNN/GRU"},
            "DIN": {"type": "Model", "desc": "Deep Interest Network with Local Activation"},
            "Transformer": {"type": "Architecture", "desc": "Self-Attention Transformer Architecture"},
            "RNN": {"type": "Architecture", "desc": "Recurrent Neural Networks for sequence modeling"},
            "Attention": {"type": "Mechanism", "desc": "Dynamic attention pooling mechanism"},
            "BPR": {"type": "Loss", "desc": "Bayesian Personalized Ranking pairwise loss"},
            "Collaborative Filtering": {"type": "Paradigm", "desc": "Interaction-based recommendation paradigm"},
            "Matrix Factorization": {"type": "Paradigm", "desc": "Latent factor decomposition SVD/NMF"},
            "Sequential Recommendation": {"type": "Paradigm", "desc": "Session/sequence-based recommendation paradigm"},
        }
        
        # Define semantic edges (relationships): (source, target) -> relation_type
        self.edges: dict[tuple[str, str], str] = {
            ("SASRec", "Transformer"): "utilizes",
            ("SASRec", "Sequential Recommendation"): "belongs_to",
            ("GRU4Rec", "RNN"): "utilizes",
            ("GRU4Rec", "Sequential Recommendation"): "belongs_to",
            ("DIN", "Attention"): "utilizes",
            ("DIN", "Collaborative Filtering"): "belongs_to",
            ("BPR", "Collaborative Filtering"): "optimizes",
            ("Matrix Factorization", "Collaborative Filtering"): "subclass_of",
            ("Sequential Recommendation", "Collaborative Filtering"): "subclass_of",
            ("Transformer", "Attention"): "contains",
        }

    def find_matching_nodes(self, query: str) -> list[str]:
        """Finds entity nodes mentioned in the user query."""
        matched = []
        query_lower = query.lower()
        for node in self.nodes:
            # Word-level/substring match for Chinese and English concepts
            if node.lower() in query_lower or (self.nodes[node]["desc"].lower() in query_lower):
                matched.append(node)
        return matched

    def expand_query(self, query: str, max_hops: int = 1) -> str:
        """Performs semantic multi-hop query expansion using the knowledge graph."""
        matched_nodes = self.find_matching_nodes(query)
        if not matched_nodes:
            return query
            
        expanded_nodes = set(matched_nodes)
        
        # Perform breadth-first traversal up to max_hops to find related concepts
        current_level = set(matched_nodes)
        for _ in range(max_hops):
            next_level = set()
            for u in current_level:
                # Traverse outgoing and incoming edges
                for (src, dst), rel in self.edges.items():
                    if src == u and dst not in expanded_nodes:
                        next_level.add(dst)
                        expanded_nodes.add(dst)
                    elif dst == u and src not in expanded_nodes:
                        next_level.add(src)
                        expanded_nodes.add(src)
            current_level = next_level
            
        # Compile related terms and append them to the original query
        expanded_terms = [node for node in expanded_nodes if node not in matched_nodes]
        if expanded_terms:
            expansion_suffix = " " + " ".join(expanded_terms)
            return query + expansion_suffix
            
        return query

    def get_relations_for_nodes(self, nodes: list[str]) -> list[dict[str, str]]:
        """Returns relation triples for rendering in Streamlit visualizations."""
        triples = []
        for (src, dst), rel in self.edges.items():
            if src in nodes or dst in nodes:
                triples.append({
                    "source": src,
                    "target": dst,
                    "relation": rel
                })
        return triples
