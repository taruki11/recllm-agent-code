from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List
from .schema import SearchResult
from .qwen_generator import QwenGenerator

logger = logging.getLogger(__name__)

class CitationVerifier:
    """Programmatic Claim-Level Citation Verifier and Hallucination Auditor."""

    def __init__(self, generator: QwenGenerator) -> None:
        self.generator = generator

    def repair_json_string(self, s: str) -> str:
        """Robustly repair typical LLM JSON syntax anomalies."""
        s = s.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

        start_idx = s.find('{')
        start_arr = s.find('[')
        if start_idx == -1 and start_arr == -1:
            return s

        if start_idx != -1 and (start_arr == -1 or start_idx < start_arr):
            end_idx = s.rfind('}')
            if end_idx != -1:
                s = s[start_idx:end_idx+1]
        else:
            end_idx = s.rfind(']')
            if end_idx != -1:
                s = s[start_arr:end_idx+1]

        result = []
        in_string = False
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if not in_string:
                if c == '"':
                    in_string = True
                result.append(c)
                i += 1
            else:
                if c == '"':
                    rest = s[i+1:].lstrip()
                    is_structural = False
                    if not rest:
                        is_structural = True
                    elif rest.startswith(',') or rest.startswith('}') or rest.startswith(']') or rest.startswith(':'):
                        if rest.startswith(','):
                            after_comma = rest[1:].lstrip()
                            if after_comma.startswith('"') or after_comma.startswith('}') or after_comma.startswith(']'):
                                is_structural = True
                        else:
                            is_structural = True

                    if is_structural:
                        in_string = False
                        result.append(c)
                        i += 1
                    else:
                        result.append('\\"')
                        i += 1
                elif c == '\\':
                    if i + 1 < n:
                        next_c = s[i+1]
                        is_latex_or_regex = False
                        if next_c in ['b', 'f', 'n', 'r', 't']:
                            if i + 2 < n and (s[i+2].isalpha() or s[i+2] in ['\\', '{', '[', '(']):
                                is_latex_or_regex = True

                        if next_c in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't'] and not is_latex_or_regex:
                            result.append('\\')
                            result.append(next_c)
                            i += 2
                        elif next_c == 'u':
                            if i + 5 < n and all(ch in '0123456789abcdefABCDEF' for ch in s[i+2:i+6]):
                                result.append('\\')
                                result.append('u')
                                result.extend(s[i+2:i+6])
                                i += 6
                            else:
                                result.append('\\\\')
                                i += 1
                        else:
                            result.append('\\\\')
                            i += 1
                    else:
                        result.append('\\\\')
                        i += 1
                elif c == '\n':
                    result.append('\\n')
                    i += 1
                elif c == '\r':
                    result.append('\\r')
                    i += 1
                elif c == '\t':
                    result.append('\\t')
                    i += 1
                else:
                    result.append(c)
                    i += 1

        repaired = "".join(result)
        repaired = re.sub(r',\s*\]', ']', repaired)
        repaired = re.sub(r',\s*\}', '}', repaired)
        repaired = re.sub(r'[\x00-\x1F\x7F]', '', repaired)
        return repaired

    def calculate_repetition_stats(self, text: str) -> dict[str, float]:
        """Compute character-level and word-level repetition rates."""
        if not text:
            return {"char_repetition_ratio": 0.0, "word_repetition_ratio": 0.0}

        # Character-level repetition ratio
        chars = [c for c in text if c.strip()]
        char_rep = 1.0 - (len(set(chars)) / len(chars)) if chars else 0.0

        # Word-level repetition ratio (handling English words and Chinese characters)
        words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
        word_rep = 1.0 - (len(set(words)) / len(words)) if words else 0.0

        return {
            "char_repetition_ratio": round(char_rep, 4),
            "word_repetition_ratio": round(word_rep, 4)
        }

    def extract_claims(self, answer: str) -> list[dict[str, Any]] | None:
        """Parses model answers into atomic claims with their mapped citation indices."""
        prompt = f"""你是一个严谨的语言学与逻辑学专家。请阅读以下【回答】，将回答拆解为若干条相互独立的“原子声明（Atomic Claims）”，并分析每条原子声明在原回答中所附带的引用标注。

【回答】：
{answer}

拆解与映射要求：
1. 每个“原子声明”必须是一个语义独立的陈述句，不要包含“并且”、“但是”等复合连词，将其拆解为最微小的原子事实。
2. 找出该声明在原回答中附带的引用序号（例如 "[1]"、"[2]" 或者是 "[No Citation]" 如果该声明完全没有引用）。
3. 如果原句同时标注了多个引用（例如 [1][2]），在 citations 列表中包含所有相关引用，如 `["[1]", "[2]"]`。

请必须只返回一个合法的 JSON 数组，不要带有 ```json 或 ``` 等 Markdown 代码块标记，也不要带有任何其他解释性文字。
数组中的每个对象格式如下：
[
  {{
    "claim": "原子声明具体内容",
    "citations": ["[1]"]
  }},
  {{
    "claim": "另一个原子声明具体内容",
    "citations": ["[No Citation]"]
  }}
]"""
        response = self.generator.generate(prompt).strip()

        # Strip potential markdown formatting
        cleaned_response = response
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        # Try standard loading
        try:
            claims = json.loads(cleaned_response)
            if isinstance(claims, dict):
                claims = [claims]
            if isinstance(claims, list):
                return claims
        except Exception:
            pass

        # Try JSON repair loading
        try:
            repaired = self.repair_json_string(response)
            claims = json.loads(repaired)
            if isinstance(claims, dict):
                claims = [claims]
            if isinstance(claims, list):
                logger.info("JSON repaired successfully!")
                return claims
        except Exception as e:
            logger.warning(f"Failed to parse and repair claim extraction JSON. Error: {e}. Raw response: {response}")

        return None

    def verify_claim(self, claim: str, context_chunk: str) -> dict[str, Any]:
        """Verify an atomic claim against its cited reference context chunk using NLI logic."""
        prompt = f"""你是一个极其严苛的 RAG 引用正确性核查专家。请仔细阅读【参考文本】，判断【待核实声明】是否真实被该参考文本所完全支持。

【参考文本】：
{context_chunk}

【待核实声明】：
{claim}

判定标准：
1. SUPPORTED: 待核实声明所包含的所有事实、概念、参数或逻辑，均可在参考文本中直接找到明确且完整的证据支持。
2. CONTRADICTED: 待核实声明中含有与参考文本事实直接相冲突、矛盾的陈述。
3. NOT_SUPPORTED: 参考文本中未提及、或仅有相关擦边词汇但没有足够且直接的论据支撑待核实声明的所有要素。

请必须只返回一个合法的 JSON 字符串，不要带有 ```json 或 ``` 等 Markdown 代码块标记，也不要带有任何其他解释性文字。
必须包含以下两个字段：
{{
  "classification": "SUPPORTED" 或 "CONTRADICTED" 或 "NOT_SUPPORTED",
  "reason": "判断的详细原因说明"
}}"""
        response = self.generator.generate(prompt).strip()

        # Strip potential markdown formatting
        cleaned_response = response
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            res = json.loads(cleaned_response)
            if "classification" not in res:
                res["classification"] = "NOT_SUPPORTED"
            if "reason" not in res:
                res["reason"] = "Missing reason in judge output."
            return res
        except Exception:
            pass

        # Try JSON repair loading
        try:
            repaired = self.repair_json_string(response)
            res = json.loads(repaired)
            if isinstance(res, list) and len(res) > 0:
                res = res[0]
            if "classification" not in res:
                res["classification"] = "NOT_SUPPORTED"
            if "reason" not in res:
                res["reason"] = "Missing reason in judge output."
            return res
        except Exception as e:
            logger.warning(f"Failed to parse and repair claim verification JSON. Error: {e}. Raw response: {response}")
            return {
                "classification": "NOT_SUPPORTED",
                "reason": f"Fallback due to parsing exception: {e}. Raw output: {response}"
            }

    def detect_repetition_collapse(self, text: str) -> bool:
        """Robustly detect repeating loop tokens or character loops representing model collapse."""
        # 1. 2-30 character sequences repeating 3+ times consecutively
        if re.search(r"(\S{2,30})\1{2,}", text):
            return True
        # 2. Single non-whitespace characters (excluding formatting punctuation) repeating 5+ times consecutively
        if re.search(r"([^\s\-*=_])\1{4,}", text):
            return True
        # 3. Mojibake/corrupted repeating patterns
        if "根据根据" in text or "蒸蒸蒸" in text or "SASASA" in text or "SASASASA" in text:
            return True

        # 4. Statistical high-ratio repetition collapse guard (for text with length > 80)
        if len(text) > 80:
            stats = self.calculate_repetition_stats(text)
            if stats["char_repetition_ratio"] > 0.85 or stats["word_repetition_ratio"] > 0.80:
                return True
        return False

    def evaluate_answer(self, answer: str, results: list[SearchResult]) -> dict[str, Any]:
        """Runs the claims verifier over a single RAG answer and compiles evaluation statistics."""
        repetition_stats = self.calculate_repetition_stats(answer)

        # Check for repetition collapse first to prevent falsifying high metric scores
        if self.detect_repetition_collapse(answer):
            logger.warning(f"Repetition collapse detected in answer: '{answer[:100]}...'. Penalizing metrics to 0.0.")
            return {
                "total_claims": 1,
                "total_cited_claims": 1,
                "supported_cited_claims": 0,
                "contradicted_claims": 1,
                "citation_precision": 0.0,
                "citation_soundness": 0.0,
                "contradiction_rate": 1.0,
                "verified_claims": [{
                    "claim": f"FAILED: Model collapsed into repetition loops. Output sample: {answer[:100]}",
                    "citations": ["[No Citation]"],
                    "classification": "CONTRADICTED",
                    "verifications": [],
                    "reasons": ["Repetition collapse detected (consecutive loops)"]
                }],
                "repetition_stats": repetition_stats,
                "parse_status": "REPETITION_COLLAPSE"
            }

        # 1. Parse into claims
        claims = self.extract_claims(answer)

        # Handle parse failure
        if claims is None:
            logger.warning("Claim extraction JSON parse failed after repair. Penalizing metrics to 0.0.")
            return {
                "total_claims": 1,
                "total_cited_claims": 1,
                "supported_cited_claims": 0,
                "contradicted_claims": 1,
                "citation_precision": 0.0,
                "citation_soundness": 0.0,
                "contradiction_rate": 1.0,
                "verified_claims": [{
                    "claim": "FAILED: Claim extraction parser failure (JSON output corrupted).",
                    "citations": ["[No Citation]"],
                    "classification": "CONTRADICTED",
                    "verifications": [],
                    "reasons": ["JSON parsing failed after repair attempts."]
                }],
                "repetition_stats": repetition_stats,
                "parse_status": "PARSE_FAILURE"
            }

        # Handle claim count abnormality (0 or >30)
        if len(claims) == 0 or len(claims) > 30:
            logger.warning(f"Abnormal claim count detected: {len(claims)}. Penalizing metrics to 0.0.")
            return {
                "total_claims": 1,
                "total_cited_claims": 1,
                "supported_cited_claims": 0,
                "contradicted_claims": 1,
                "citation_precision": 0.0,
                "citation_soundness": 0.0,
                "contradiction_rate": 1.0,
                "verified_claims": [{
                    "claim": f"FAILED: Abnormal claim count ({len(claims)} claims generated). Likely generation hallucination or repeat loop.",
                    "citations": ["[No Citation]"],
                    "classification": "CONTRADICTED",
                    "verifications": [],
                    "reasons": [f"Claim count abnormal: {len(claims)} claims (limit: 1-30)."]
                }],
                "repetition_stats": repetition_stats,
                "parse_status": "ABNORMAL_CLAIM_COUNT"
            }

        # Map citations to their actual text contexts
        # e.g. results are top-K, usually index mapped as [1] -> results[0], [2] -> results[1]...
        context_map = {}
        for idx, r in enumerate(results, 1):
            context_map[f"[{idx}]"] = r.chunk.text

        verified_claims = []
        supported_with_cit = 0
        total_with_cit = 0
        contradicted_count = 0

        for item in claims:
            claim_text = item.get("claim", "")
            citations = item.get("citations", [])

            if not claim_text:
                continue

            # Remove any empty list or invalid elements
            citations = [c for c in citations if c]
            if not citations:
                citations = ["[No Citation]"]

            results_list = []
            has_valid_citation = False

            for cit in citations:
                if cit in context_map:
                    has_valid_citation = True
                    chunk_text = context_map[cit]
                    ver = self.verify_claim(claim_text, chunk_text)
                    results_list.append({
                        "citation": cit,
                        "classification": ver["classification"],
                        "reason": ver["reason"]
                    })
                elif cit == "[No Citation]":
                    # Non-cited claim verification (against the union of all contexts to check if it's hidden hallucination)
                    all_contexts = "\n\n".join([r.chunk.text for r in results])
                    ver = self.verify_claim(claim_text, all_contexts)
                    results_list.append({
                        "citation": "[No Citation]",
                        "classification": ver["classification"],
                        "reason": ver["reason"]
                    })

            # Aggregate classification for this claim
            # If any cited source supports it, it's SUPPORTED.
            # If any cited source contradicts it and none supports it, it's CONTRADICTED.
            # Otherwise NOT_SUPPORTED.
            final_class = "NOT_SUPPORTED"
            reasons = []
            for r_res in results_list:
                reasons.append(f"{r_res['citation']}: {r_res['classification']} ({r_res['reason']})")
                if r_res["classification"] == "SUPPORTED":
                    final_class = "SUPPORTED"
                    break

            if final_class != "SUPPORTED":
                if any(r_res["classification"] == "CONTRADICTED" for r_res in results_list):
                    final_class = "CONTRADICTED"

            # Count metrics
            if has_valid_citation:
                total_with_cit += 1
                if final_class == "SUPPORTED":
                    supported_with_cit += 1

            if final_class == "CONTRADICTED":
                contradicted_count += 1

            verified_claims.append({
                "claim": claim_text,
                "citations": citations,
                "classification": final_class,
                "verifications": results_list,
                "reasons": reasons
            })

        # Calculate metrics
        total_claims = len(claims) if len(claims) > 0 else 1
        citation_precision = supported_with_cit / total_with_cit if total_with_cit > 0 else 1.0
        citation_soundness = supported_with_cit / total_claims
        contradiction_rate = contradicted_count / total_claims

        return {
            "total_claims": len(claims),
            "total_cited_claims": total_with_cit,
            "supported_cited_claims": supported_with_cit,
            "contradicted_claims": contradicted_count,
            "citation_precision": round(citation_precision, 4),
            "citation_soundness": round(citation_soundness, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "verified_claims": verified_claims,
            "repetition_stats": repetition_stats,
            "parse_status": "SUCCESS"
        }
