from __future__ import annotations
import json
import re
from typing import List, Dict, Any
from .qwen_generator import QwenGenerator


VALID_PLAN_TOOLS = {
    "retrieve_rec_docs",
    "retrieve_project_reports",
    "compare_resume_jd",
    "generate_mock_questions",
    "web_search",
}


def required_tools_for_query(query: str) -> List[str]:
    """Return high-confidence tools in deterministic execution order."""
    needs_mock = any(keyword in query for keyword in ("面试题", "模拟题", "真题", "考点", "出题", "追问题"))
    needs_resume = any(keyword in query for keyword in ("简历", "JD", "岗位匹配", "匹配度", "求职"))
    needs_project = any(keyword in query for keyword in (
        "项目", "项目报告", "评估报告", "实验报告", "实验日志", "当前统一", "当前仓库",
        "RecLLM-Agent", "GenAdRec", "RRF", "3060", "SFT v", "DPO v",
    ))
    explicit_domain_lookup = any(keyword in query for keyword in (
        "先检索领域资料", "领域资料", "推荐系统资料",
    ))
    domain_markers = (
        "DIN", "DIEN", "SASRec", "BERT4Rec", "MMOE", "GraphSAGE", "知识图谱",
        "序列推荐", "推荐系统", "双塔", "召回", "排序", "曝光偏差",
    )
    needs_rec = (
        explicit_domain_lookup
        or (not needs_project and not needs_resume and any(marker in query for marker in domain_markers))
        or (needs_project and "曝光偏差" in query)
    )

    tools = []
    if needs_rec:
        tools.append("retrieve_rec_docs")
    if needs_project:
        tools.append("retrieve_project_reports")
    if needs_resume:
        tools.append("compare_resume_jd")
    if needs_mock:
        tools.append("generate_mock_questions")
    return tools[:3]


def normalize_plan(query: str, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Constrain an LLM plan with high-confidence intent rules and deduplication."""
    sanitized = [
        step for step in plan
        if isinstance(step, dict) and step.get("tool") in VALID_PLAN_TOOLS
    ]
    required = required_tools_for_query(query)
    selected_tools = required or [
        step["tool"] for step in sanitized
        if step["tool"] not in {"none"} and step["tool"] in VALID_PLAN_TOOLS
    ]

    deduped_tools = []
    for tool in selected_tools:
        if tool not in deduped_tools:
            deduped_tools.append(tool)
    deduped_tools = deduped_tools[:3]

    if not deduped_tools:
        return [{
            "step": 1,
            "thought": "无需调用专用工具，直接回答",
            "tool": "none",
            "input": query,
        }]

    normalized = []
    for index, tool in enumerate(deduped_tools, 1):
        matching_step = next((step for step in sanitized if step.get("tool") == tool), None)
        normalized.append({
            "step": index,
            "thought": f"受约束计划：执行 {tool}",
            "tool": tool,
            "input": str((matching_step or {}).get("input") or query),
        })
    return normalized


class Planner:
    """Planning module to decompose complex recommendation queries into sub-tasks.
    Aligns with DeepLearning.AI Pattern 3: Planning."""
    def __init__(self, generator: QwenGenerator) -> None:
        self.generator = generator

    def plan(self, query: str) -> List[Dict[str, Any]]:
        # Clean query: remove problematic chars for Windows GBK
        query = query.replace('�', '').strip()
        query = query.replace('"', '"').replace('"', '"')
        query = query.replace(''', "'").replace(''', "'")

        # Build prompt (use .replace() to safely insert query without f-string brace issues)
        prompt_template = """你是一个专业的推荐系统智能体任务规划专家。你需要判断用户的提问是否复杂，是否需要拆解为多个子任务或子查询来解决。

可选工具列表：
1. `retrieve_rec_docs(query)`: 检索推荐系统算法、理论、模型设计等学术或技术概念文档（适合"DIN和DIEN的区别"、"SASRec原理"）。
2. `retrieve_project_reports(query)`: 检索推荐系统项目实践、简历评估日志、GenAdRec实验报告等文档（适合工程报告、评估指标、实验参数）。
3. `compare_resume_jd(resume_jd_text)`: 深度评估简历与岗位匹配度并提供优化建议。
4. `generate_mock_questions(topic)`: 生成特定搜推算法（如SASRec/DIN）的大厂面试真题及核心考点。
5. `web_search(query)`: 联网搜索实时信息（适合"最新论文"、"vs对比"、"薪资待遇"）。

拆解原则：
- 如果用户的提问是单一、简单的问题（例如："DIN和DIEN有什么区别"），无需拆解，直接返回一个步骤。
- 如果提问是复合的（例如："请先帮我检索SASRec的工作原理，然后针对它给我出三道面试题"），请将其拆解为多个步骤，前一步检索完后，下一步再做生成。
- 如果需要比对或诊断项目实验（例如："结合我的项目报告，看看为什么GenAdRec的HR@10指标很低"），请拆解为：步骤1检索项目报告；步骤2结合观察生成诊断分析。
- 如果提问是"对比A和B"、"A vs B"类问题，请拆解为两个独立搜索步骤，分别搜索A和B的信息，然后在最后一步整合对比。

Few-Shot 示例：

示例1（对比类问题）：
用户输入："Qwen2.5 和 Llama3 模型对比如何？"
输出：
[
  {"step": 1, "thought": "先搜索 Qwen2.5 模型的信息", "tool": "web_search", "input": "Qwen2.5 模型 参数 性能"},
  {"step": 2, "thought": "再搜索 Llama3 模型的信息", "tool": "web_search", "input": "Llama3 模型 参数 性能"},
  {"step": 3, "thought": "整合两次搜索结果，生成对比分析", "tool": "none", "input": "整合 Qwen2.5 和 Llama3 的对比分析"}
]

示例2（单一问题）：
用户输入："SASRec 的原理是什么？"
输出：
[
  {"step": 1, "thought": "直接检索 SASRec 的原理文档", "tool": "retrieve_rec_docs", "input": "SASRec 原理"}
]

示例3（复合问题）：
用户输入："请先帮我检索 SASRec 的工作原理，然后针对它给我出三道面试题"
输出：
[
  {"step": 1, "thought": "先检索 SASRec 的工作原理", "tool": "retrieve_rec_docs", "input": "SASRec 工作原理"},
  {"step": 2, "thought": "根据检索结果生成面试题", "tool": "generate_mock_questions", "input": "SASRec"}
]

请必须只返回一个 JSON 数组格式（确保格式完全合法，不要包含任何 markdown 块标记，也不要有任何其他解释性文字）：
[
  {"step": 1, "thought": "拆解本步的推理思考", "tool": "选用的工具（必须是 retrieve_rec_docs, retrieve_project_reports, compare_resume_jd, generate_mock_questions, web_search 之一，或 none）", "input": "传递给该工具的具体子查询/参数"}
]

用户问题：{query}

任务规划 JSON："""

        # Replace {query} placeholder (using {{}} for literal braces in template)
        prompt = prompt_template.replace("{query}", query)

        response = self.generator.generate(prompt).strip()

        # Clean markdown fences
        cleaned = response
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Basic cleanup
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r',\s*\}', '}', cleaned)

        try:
            plan_list = json.loads(cleaned)
            if isinstance(plan_list, list) and len(plan_list) > 0:
                return normalize_plan(query, plan_list)
        except Exception:
            # Attempt regex find
            json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
            if json_match:
                try:
                    plan_list = json.loads(json_match.group(0))
                    if isinstance(plan_list, list) and len(plan_list) > 0:
                        return normalize_plan(query, plan_list)
                except Exception:
                    pass

        # Fallback to single step none
        return normalize_plan(query, [{
            "step": 1,
            "thought": "无法进行复杂拆解，降级为单步直接回答",
            "tool": "none",
            "input": query
        }])

    def plan_tot(self, query: str, n_paths: int = 3) -> List[Dict[str, Any]]:
        """
        Tree-of-Thought (LLM-driven): Generate multiple planning paths using LLM,
        then evaluate and select the best path.

        Optimized: Uses batch inference (single forward pass for all paths).

        Args:
            query: User query
            n_paths: Number of paths to generate (default: 3)

        Returns:
            Best plan (list of steps, each step is a Dict)
        """
        # Step 1: Build all N prompts
        prompts = []

        for i in range(n_paths):
            # Build prompt for plan generation
            prompt = f"""你是一个专业的推荐系统智能体任务规划专家。你需要为用户的提问生成一个或多个步骤的任务规划。

可选工具列表：
1. `retrieve_rec_docs(query)`: 检索推荐系统算法、理论、模型设计等学术或技术概念文档。
2. `retrieve_project_reports(query)`: 检索推荐系统项目实践、简历评估日志、实验报告等文档。
3. `compare_resume_jd(resume_jd_text)`: 深度评估简历与岗位匹配度并提供优化建议。
4. `generate_mock_questions(topic)`: 生成特定搜推算法（如SASRec/DIN）的大厂面试真题及核心考点。
5. `web_search(query)`: 联网搜索实时信息（适合"最新论文"、"vs对比"、"薪资待遇"）。

请生成一个任务规划（JSON数组格式），包含1-3个步骤。每个步骤包含：
- "step": 步骤编号
- "thought": 本步的推理思考
- "tool": 选用的工具（必须是 retrieve_rec_docs, retrieve_project_reports, compare_resume_jd, generate_mock_questions, web_search 之一，或 none）
- "input": 传递给该工具的具体子查询/参数

用户问题：{query}

任务规划 JSON（只返回 JSON 数组，不要有其他文字）："""

            # Add diversity instruction for i > 0
            if i > 0:
                prompt += f"\n\n请生成一个不同的规划方案（方案 {i+1}），尝试不同的工具组合或步骤顺序。"

            prompts.append(prompt)

        # Step 2: Batch generate all N plans at once (single forward pass)
        # Note: batch inference uses same temperature (0.7) for all samples
        # Diversity comes from different prompts + sampling
        responses = self.generator.generate_batch(
            prompts,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )

        # Step 3: Parse each response
        paths = []
        for i, response in enumerate(responses):
            response = response.strip()

            # Clean markdown fences
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Basic cleanup
            response = re.sub(r',\s*\]', ']', response)
            response = re.sub(r',\s*\}', '}', response)

            try:
                plan_list = json.loads(response)
                if isinstance(plan_list, list) and len(plan_list) > 0:
                    paths.append(plan_list)
            except Exception:
                # Attempt regex find
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    try:
                        plan_list = json.loads(json_match.group(0))
                        if isinstance(plan_list, list) and len(plan_list) > 0:
                            paths.append(plan_list)
                    except Exception:
                        pass

        # Step 4: If no valid paths generated, fallback to rule-based single path
        if not paths:
            return [{
                "step": 1,
                "thought": "无法进行复杂拆解，降级为单步直接回答",
                "tool": "none",
                "input": query
            }]

        # Step 5: Evaluate each path using LLM (still serial, but shorter)
        scored_paths = []
        for path in paths:
            # Build evaluation prompt
            eval_prompt = f"""你是一个任务规划评估专家。请评估以下任务规划的质量（1-10分）。

用户问题：{query}

任务规划：
{json.dumps(path, ensure_ascii=False, indent=2)}

评估标准：
1. 工具选择是否合适（是否选择了正确的工具来完成任务）
2. 步骤顺序是否合理（是否按照正确的顺序执行步骤）
3. 是否完整地解决了用户问题（是否覆盖了所有必要的子任务）

请只返回一个 JSON 对象，包含 "score" 字段（1-10 的整数）：
{{"score": 8}}"""

            eval_response = self.generator.generate(eval_prompt, max_new_tokens=128).strip()

            # Clean and parse evaluation response
            eval_cleaned = eval_response
            if eval_cleaned.startswith("```json"):
                eval_cleaned = eval_cleaned[7:]
            if eval_cleaned.startswith("```"):
                eval_cleaned = eval_cleaned[3:]
            if eval_cleaned.endswith("```"):
                eval_cleaned = eval_cleaned[:-3]
            eval_cleaned = eval_cleaned.strip()

            try:
                eval_result = json.loads(eval_cleaned)
                score = eval_result.get("score", 5)
                scored_paths.append((path, score))
            except Exception:
                # If evaluation fails, assign default score
                scored_paths.append((path, 5))

        # Step 6: Select the best path (highest score)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        best_plan = scored_paths[0][0]

        return best_plan
