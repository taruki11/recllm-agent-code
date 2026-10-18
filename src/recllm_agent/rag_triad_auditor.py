from __future__ import annotations

import json
import logging
from typing import Any
from .qwen_generator import QwenGenerator

logger = logging.getLogger(__name__)

class RAGTriadAuditor:
    """Advanced Agentic Auditor running RAG Triad evaluations (Context Relevance, Faithfulness, Answer Relevance)."""
    
    def __init__(self, generator: QwenGenerator) -> None:
        self.generator = generator

    def evaluate_context_relevance(self, query: str, context: str) -> dict[str, Any]:
        """Evaluates how relevant the retrieved context chunks are to the user query."""
        prompt = f"""你是一个高级的检索品质审计专家。请阅读以下【用户查询】和【检索到的上下文】，评估这些上下文是否包含回答查询所需的关键信息。
        
【用户查询】：
{query}

【检索到的上下文】：
{context}

判定标准：
1. 输出一个 0.0 到 1.0 之间的浮点数评分 (score)，代表检索上下文与查询的语义关联度和信息覆盖度（1.0 代表完美包含所需全部信息，0.0 代表完全无关）。
2. 提供详细且深刻的评价理由 (reason)。

请必须只返回一个合法的 JSON 字符串，不要带有 ```json 或 ``` 等 Markdown 代码块标记，也不要带有任何其他解释性文字。
格式如下：
{{
  "score": 0.85,
  "reason": "上下文包含了关于该算法的核心公式和设计，但缺少了参数微调的实战案例..."
}}"""
        response = self.generator.generate(prompt).strip()
        return self._safe_parse_output(response, "Context Relevance")

    def evaluate_faithfulness(self, context: str, answer: str) -> dict[str, Any]:
        """Evaluates faithfulness / groundedness: check if all assertions in the answer are supported by the context."""
        prompt = f"""你是一个严苛的真实性与幻觉审计专家。请阅读以下【参考上下文】和【生成的回答】，评估回答中的每一个核心声明，是否都可以在参考上下文中找到坚实且明确的事实证据支撑（不包含任何脑补或外部知识）。
        
【参考上下文】：
{context}

【生成的回答】：
{answer}

判定标准：
1. 输出一个 0.0 到 1.0 之间的浮点数评分 (score)，代表回答的真实与接地程度（1.0 代表所有观点100%均有坚实的上下文事实背书，0.0 代表全是捏造或脑补的幻觉）。
2. 提供详细且深刻的评价理由 (reason)。

请必须只返回一个合法的 JSON 字符串，不要带有 ```json 或 ``` 等 Markdown 代码块标记，也不要带有任何其他解释性文字。
格式如下：
{{
  "score": 0.95,
  "reason": "回答严格基于文献进行了总结，唯一美中不足是对某个参数进行了细微的外部推论，属于轻微的过度演绎..."
}}"""
        response = self.generator.generate(prompt).strip()
        return self._safe_parse_output(response, "Faithfulness")

    def evaluate_answer_relevance(self, query: str, answer: str) -> dict[str, Any]:
        """Evaluates answer relevance: checks if the generated response directly addresses the user question."""
        prompt = f"""你是一个资深的问答品质审计专家。请阅读以下【用户查询】和【生成的回答】，评估该回答是否直接、切中要害地回答了用户问题，有没有包含多余无关的杂音、或者避重就轻。
        
【用户查询】：
{query}

【生成的回答】：
{answer}

判定标准：
1. 输出一个 0.0 到 1.0 之间的浮点数评分 (score)，代表回答对查询的直接解决度和契合度（1.0 代表完美直接解答且结构清晰，0.0 代表完全答非所问、离题万里）。
2. 提供详细且深刻的评价理由 (reason)。

请必须只返回一个合法的 JSON 字符串，不要带有 ```json 或 ``` 等 Markdown 代码块标记，也不要带有任何其他解释性文字。
格式如下：
{{
  "score": 0.90,
  "reason": "回答结构分明，完美解答了核心数学机制，但如果在首段能直接给出定义，体验会更棒..."
}}"""
        response = self.generator.generate(prompt).strip()
        return self._safe_parse_output(response, "Answer Relevance")

    def run_full_triad_audit(self, query: str, context: str, answer: str) -> dict[str, Any]:
        """Runs all three dimensions of the RAG Triad and compiles the results."""
        cr = self.evaluate_context_relevance(query, context)
        ft = self.evaluate_faithfulness(context, answer)
        ar = self.evaluate_answer_relevance(query, answer)
        
        avg_score = (cr["score"] + ft["score"] + ar["score"]) / 3.0
        
        return {
            "context_relevance": cr,
            "faithfulness": ft,
            "answer_relevance": ar,
            "triad_average": round(avg_score, 4)
        }

    def _safe_parse_output(self, response: str, dimension_name: str) -> dict[str, Any]:
        """Gracefully strips markdown blocks and parses JSON output with robust fallbacks."""
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            res = json.loads(text)
            if "score" not in res:
                res["score"] = 0.5
            if "reason" not in res:
                res["reason"] = "No reason provided by judge."
            res["score"] = float(res["score"])
        except Exception as e:
            logger.warning(f"Failed to parse RAG Triad JSON for {dimension_name}: {e}. Output: {response}")
            res = {
                "score": 0.5,
                "reason": f"Fallback due to parsing error: {e}. Raw response: {response[:100]}"
            }
        return res
