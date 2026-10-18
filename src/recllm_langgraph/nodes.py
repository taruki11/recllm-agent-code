# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import re
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from recllm_langgraph.state import AgentState
from recllm_agent.schema import Chunk, SearchResult

RETRIEVAL_TOOLS = {"retrieve_rec_docs", "retrieve_project_reports", "web_search"}


def _requested_question_count(query: str, default: int = 3) -> int:
    """Extract an explicit interview-question count from the original user query."""
    digit_match = re.search(r"(\d+)\s*道", query)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), 10))
    for marker, count in {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}.items():
        if f"{marker}道" in query:
            return count
    return default


def _try_empty_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _execute_single_tool(
    tool_name: str,
    tool_input: str,
    agent_tools,
    memory,
    query: str = ""
) -> str:
    """
    Execute a single tool and return the observation string.
    Also caches retrieved chunks into memory for citation audit.
    """
    observation = "没有获取到有效观察结果。"
    if tool_name == "none" or not tool_name:
        return observation

    try:
        if tool_name == "compare_resume_jd":
            cached_resume = memory.get_user_profile("resume")
            if cached_resume:
                comparison_text = f"【候选人简历】：\n{cached_resume}\n\n【匹配岗位JD描述】：\n{tool_input}"
                tool_res = agent_tools.compare_resume_jd(comparison_text, "推荐系统算法工程师岗")
            else:
                tool_res = agent_tools.compare_resume_jd(tool_input, "推荐系统算法工程师岗")
            observation = tool_res.get("comparison", "")

        elif tool_name in ["retrieve_rec_docs", "retrieve_project_reports"]:
            if tool_name == "retrieve_rec_docs":
                tool_res = agent_tools.retrieve_rec_docs(tool_input)
            else:
                tool_res = agent_tools.retrieve_project_reports(tool_input)

            chunks_dicts = tool_res.get("results", [])
            results_text = []
            temp_results = []
            for idx, cd in enumerate(chunks_dicts, 1):
                title = cd["chunk"]["title"]
                text = cd["chunk"]["text"]
                results_text.append(f"[{idx}] {title}: {text[:350]}")
                ch = Chunk.from_dict(cd["chunk"])
                sr = SearchResult(chunk=ch, score=cd.get("score", 1.0), rank=idx)
                temp_results.append(sr)

            memory.cache_context(temp_results)
            observation = "\n\n".join(results_text) if results_text else "检索库中没有找到相关匹配条目。"

        elif tool_name == "generate_mock_questions":
            count = _requested_question_count(query)
            tool_res = agent_tools.generate_mock_questions(tool_input, count=count)
            observation = tool_res.get("mock_questions", "")

        elif tool_name == "web_search":
            tool_res = agent_tools.web_search(tool_input)
            chunks_dicts = tool_res.get("results", [])
            results_text = []
            temp_results = []
            for idx, cd in enumerate(chunks_dicts, 1):
                title = cd["chunk"]["title"]
                text = cd["chunk"]["text"]
                url = cd["chunk"].get("source_path", "")
                results_text.append(f"[{idx}] {title}: {text[:350]} (来源: {url})")
                ch = Chunk.from_dict(cd["chunk"])
                sr = SearchResult(chunk=ch, score=cd.get("score", 1.0), rank=idx)
                temp_results.append(sr)

            memory.cache_context(temp_results)
            observation = "\n\n".join(results_text) if results_text else "联网搜索未找到相关信息。"

        else:
            observation = f"未知工具: {tool_name}"

    except Exception as e:
        observation = f"工具执行出错: {str(e)}"

    return observation


def _find_reusable_tool_result(
    tool_results: List[Dict[str, Any]],
    tool_name: str,
    tool_input: str,
) -> Dict[str, Any] | None:
    """Reuse an identical successful tool call within the same workflow run."""
    for result in tool_results:
        if (
            result.get("success", True)
            and result.get("tool") == tool_name
            and result.get("input") == tool_input
        ):
            return result
    return None


def _format_tool_native_outputs(tool_results: List[Dict[str, Any]]) -> str:
    """Format successful non-retrieval tool outputs without regenerating them."""
    sections = []
    for result in tool_results:
        tool_name = result.get("tool", "")
        output = str(result.get("output") or "").strip()
        if (
            result.get("success", True)
            and tool_name
            and tool_name not in RETRIEVAL_TOOLS
            and tool_name != "none"
            and output
        ):
            sections.append(f"### {tool_name}\n\n{output}")
    return "\n\n".join(sections)


def _compose_answer(grounded_answer: str, tool_native_answer: str) -> str:
    grounded_answer = grounded_answer.strip()
    tool_native_answer = tool_native_answer.strip()
    if grounded_answer and tool_native_answer:
        return f"{grounded_answer}\n\n## 工具生成结果\n\n{tool_native_answer}"
    return grounded_answer or tool_native_answer


def _extractive_grounded_fallback(context: str, limit: int = 2) -> str:
    """Return a conservative cited fallback when generation drops every citation."""
    matches = re.findall(r"(?ms)^\s*\[(\d+)\]\s+(.*?)(?=^\s*\[\d+\]\s+|\Z)", context)
    bullets = []
    for number, text in matches[:limit]:
        compact = re.sub(r"\s+", " ", text).strip()
        if compact:
            bullets.append(f"- {compact[:280]} [{number}]")
    if not bullets:
        return "根据现有检索资料无法形成可验证回答。"
    return (
        "为避免引入无依据内容，以下仅列出检索资料可直接确认的信息：\n"
        + "\n".join(bullets)
        + "\n\n超出以上资料范围的结论仍需补充证据。"
    )

def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 1: Multi-Step Task Planning. Decomposes the user query into steps."""
    planner = config["configurable"]["planner"]
    query = state["query"]

    start = time.time()
    steps_plan = planner.plan(query)
    latency = time.time() - start

    plan_details = " | ".join([
        f"Step {s.get('step', 1)}: {s.get('thought', '')} -> {s.get('tool', 'none')}({s.get('input', '')})"
        for s in steps_plan
    ])

    trace_entry = {
        "step": "Planner Deconstruct",
        "status": "SUCCESS",
        "details": f"Generated plan: {plan_details}",
        "latency_s": latency
    }

    return {
        "plan": steps_plan,
        "trace": state.get("trace", []) + [trace_entry]
    }

def router_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 2: Intent Routing. Chooses the target tool for execution."""
    router = config["configurable"]["router"]
    query = state["query"]

    start = time.time()
    selected_route = router.route(query)
    latency = time.time() - start

    trace_entry = {
        "step": "Intent Routing",
        "status": "SUCCESS",
        "details": f"Selected route: `{selected_route}`",
        "latency_s": latency
    }

    return {
        "route": selected_route,
        "trace": state.get("trace", []) + [trace_entry]
    }

def tool_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 3: Tool Execution. Executes a SINGLE step (ReAct style).

    Instead of executing all steps at once, we execute one step at a time,
    then let react_decision_node decide whether to continue or finish.
    """
    agent_tools = config["configurable"]["agent_tools"]
    memory = config["configurable"]["memory"]
    route = state["route"]
    query = state["query"]
    plan = state.get("plan", [])
    react_step_index = state.get("react_step_index", 0)
    tool_results = list(state.get("tool_results", []))

    # ── ReAct: Single-step execution ──────────────────────────────
    if len(plan) > 1:
        # Execute ONLY the current step (react_step_index)
        if react_step_index >= len(plan):
            # All steps completed, skip tool execution
            return {
                "tool_results": tool_results,
                "retrieved_context": memory.get_cached_context_str(limit=6),
                "trace": state.get("trace", [])
            }

        step = plan[react_step_index]
        tool_name = step.get("tool", "none")
        tool_input = step.get("input", query)
        reusable = _find_reusable_tool_result(tool_results, tool_name, tool_input)
        if reusable is not None:
            trace_entry = {
                "step": f"Tool Execution (Step {react_step_index+1}/{len(plan)})",
                "status": "REUSED",
                "details": f"Reused identical successful call `{tool_name}` with input `{tool_input}`.",
                "latency_s": 0.0,
            }
            return {
                "tool_results": tool_results,
                "retrieved_context": memory.get_cached_context_str(limit=6),
                "react_step_index": react_step_index + 1,
                "react_history": list(state.get("react_history", [])),
                "trace": state.get("trace", []) + [trace_entry],
            }

        start = time.time()
        observation = _execute_single_tool(
            tool_name, tool_input, agent_tools, memory, query
        )
        tool_succeeded = not observation.startswith(("工具执行出错:", "未知工具:"))
        latency = time.time() - start
        _try_empty_vram()

        # Append to tool_results
        tool_results.append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation,
            "success": tool_succeeded,
        })

        # Append to react_history
        react_history = list(state.get("react_history", []))
        react_history.append({
            "thought": step.get("thought", ""),
            "action": tool_name,
            "action_input": tool_input,
            "observation": observation[:500]  # Truncate for history
        })

        trace_entry = {
            "step": f"Tool Execution (Step {react_step_index+1}/{len(plan)})",
            "status": "SUCCESS" if tool_succeeded else "ERROR",
            "details": f"Executed tool `{tool_name}` with input `{tool_input}`. Observation length: {len(observation)} chars.",
            "latency_s": latency
        }

        return {
            "tool_results": tool_results,
            "retrieved_context": memory.get_cached_context_str(limit=6),
            "react_step_index": react_step_index + 1,
            "react_history": react_history,
            "trace": state.get("trace", []) + [trace_entry]
        }

    # ── Single-step execution (original logic for plan <= 1) ─────────
    if route == "none":
        return {
            "tool_results": tool_results,
            "retrieved_context": "",
            "react_step_index": max(react_step_index + 1, len(plan) if plan else 1),
        }

    start = time.time()

    # Try to find step-specific input from planner results
    tool_input = query
    for step_info in plan:
        if step_info.get("tool") == route:
            tool_input = step_info.get("input", query)
            break

    reusable = _find_reusable_tool_result(tool_results, route, tool_input)
    if reusable is not None:
        return {
            "tool_results": tool_results,
            "retrieved_context": memory.get_cached_context_str(limit=6),
            "react_step_index": max(react_step_index + 1, len(plan) if plan else 1),
            "trace": state.get("trace", []) + [{
                "step": f"Tool Execution ({route})",
                "status": "REUSED",
                "details": f"Reused identical successful call `{route}` with input `{tool_input}`.",
                "latency_s": 0.0,
            }],
        }

    observation = _execute_single_tool(route, tool_input, agent_tools, memory, query)
    tool_succeeded = not observation.startswith(("工具执行出错:", "未知工具:"))
    results_list = tool_results + [{
        "tool": route,
        "input": tool_input,
        "output": observation,
        "success": tool_succeeded,
    }]
    latency = time.time() - start
    _try_empty_vram()

    retrieval_tools = {"retrieve_rec_docs", "retrieve_project_reports", "web_search"}
    if route in retrieval_tools and observation.strip():
        retrieved_context = observation
    elif route in retrieval_tools:
        retrieved_context = memory.get_cached_context_str(limit=6)
    else:
        retrieved_context = ""

    trace_entry = {
        "step": f"Tool Execution ({route})",
        "status": "SUCCESS" if tool_succeeded else "ERROR",
        "details": f"Executed tool `{route}` with input `{tool_input}`. Observation length: {len(observation)} chars.",
        "latency_s": latency
    }

    return {
        "tool_results": results_list,
        "retrieved_context": retrieved_context,
        "react_step_index": max(react_step_index + 1, len(plan) if plan else 1),
        "trace": state.get("trace", []) + [trace_entry]
    }


def answer_writer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 4: Answer Synthesis. Writes the draft response."""
    generator = config["configurable"]["generator"]
    memory = config["configurable"]["memory"]
    query = state["query"]
    context_str = state.get("retrieved_context", "")
    tool_results = state.get("tool_results", [])
    tool_native_answer = _format_tool_native_outputs(tool_results)

    # Reconstruct observation history from tool results
    observation_history = ""
    retrieval_observation_history = ""
    for idx, tr in enumerate(tool_results, 1):
        tool_name = tr["tool"]
        obs = tr["output"]
        observation_history += f"\n[Step {idx} Tool: {tool_name} Observation]:\n{obs[:600]}\n\n"
        if tool_name in RETRIEVAL_TOOLS:
            retrieval_observation_history += (
                f"\n[Step {idx} Tool: {tool_name} Observation]:\n{obs[:600]}\n\n"
            )

    history_summary = memory.get_history_summary()

    # Use different prompt depending on whether we have retrieval context
    if not context_str.strip() and state.get("tool_results"):
        prompt = f"""你是一个有用的AI助手。请根据已经执行完成的工具输出回答用户问题。

【工具输出】：
{observation_history}

【用户问题】：
{query}

要求：
1. 优先使用工具输出，不要忽略工具已经完成的工作
2. 不编造工具输出中不存在的用户经历、评分或事实
3. 非检索工具输出不要求添加引用编号
4. 回答简洁、准确、有用

请开始回答："""
    elif not context_str.strip():
        # No tool evidence: let the model answer from its own knowledge.
        prompt = f"""你是一个有用的AI助手，请用专业、客观、严谨的态度回答用户的问题。

【会话历史】：
{history_summary}

【用户问题】：
{query}

要求：
1. 用自己的知识回答，不需要引用资料
2. 请用你自身的知识直接回答，禁止用"我没有信息"、"我没有获取到"等说辞作为开头
3. 请尽最大努力回答，只有当你确实完全不知道答案时，才说"我不知道"
4. 严禁编造虚假信息或夸大表达
5. 回答要简洁、准确、有用

请开始回答："""
    else:
        # Has retrieval context: use citation prompt (strict)
        prompt = f"""你是一个高级的推荐系统垂直智能体专家。请综合用户的提问、我们多步规划执行所得到的参考事实，以及会话的历史记忆，给出最为专业、客观且严谨的终极回答。

【会话历史问答概要】：
{history_summary}

【多步工具调用所得事实资料】：
{context_str}

【多步工具调用所得观察记录】：
{retrieval_observation_history}

【当前用户的问题】：
{query}

⚠️ 回答质量要求（必须满足，否则审计不及格）：
1. 【强制引用】每一个关键论点或事实结论，必须使用 [1] [2] 等标注其在"多步工具调用所得事实资料"中的序号来源。
   - 示例：推荐系统存在冷启动问题 [1]，业界常用基于内容的推荐来缓解 [2]。
   - 如果某句话来自资料 [3]，必须紧跟 [3]，不能省略。
2. 【完全基于资料】切勿编造虚假信息。所有回答内容必须能在上面【多步工具调用所得事实资料】中找到依据。
3. 【严禁绝对化表述】禁止使用"完美解决"、"100%成功"、"完全取代"等过度武断的词汇。
4. 【实在无法回答时】如果事实资料中确实没有相关信息，请明确说"根据现有资料无法回答"，不要假装知道。

⚠️ 重要提醒：
- 引用必须准确且必要，不要为了增加引用数量而重复或滥用引用。
- 如果回答中完全没有 [1] [2] 引用标记，将被判定为"不依据资料"，审计直接不及格。
- 当资料不足时，缩小回答范围并明确说明边界。
{"- 只回答检索资料能够支持的事实部分；其他工具已经生成的内容会由系统原样附加，不要复述或重新生成。" if tool_native_answer else ""}

请开始回答："""

    if context_str.strip():
        prompt = f"""请只依据下面的编号资料回答用户问题。

编号资料：
{context_str}

用户问题：
{query}

要求：
1. 用简洁的中文分点回答，优先覆盖问题的每个子问题。
2. 每条事实后必须紧跟对应引用编号，例如 [1]。
3. 不补充资料之外的事实；资料不足时明确说明边界。
4. 不复述模拟题等非检索工具输出，系统会原样附加。

回答："""

    start = time.time()
    _try_empty_vram()
    raw_generated_answer = generator.generate(prompt).strip()
    generated_answer = raw_generated_answer
    grounding_fallback_used = False
    if context_str.strip() and not re.search(r"\[\d+\]", generated_answer):
        generated_answer = _extractive_grounded_fallback(context_str)
        grounding_fallback_used = True
    latency = time.time() - start

    if context_str.strip():
        raw_grounded_answer = raw_generated_answer
        grounded_answer = generated_answer
        final_tool_native_answer = tool_native_answer
        grounding_strategy = "extractive_fallback" if grounding_fallback_used else "model_cited"
    elif tool_results:
        raw_grounded_answer = ""
        grounded_answer = ""
        final_tool_native_answer = generated_answer
        grounding_strategy = "tool_native"
    else:
        raw_grounded_answer = raw_generated_answer
        grounded_answer = generated_answer
        final_tool_native_answer = ""
        grounding_strategy = "model_knowledge"
    answer = _compose_answer(grounded_answer, final_tool_native_answer)

    trace_entry = {
        "step": "Answer Formulation",
        "status": "SUCCESS",
        "details": "Generated final answer draft.",
        "latency_s": latency
    }

    return {
        "answer": answer,
        "raw_grounded_answer": raw_grounded_answer,
        "grounded_answer": grounded_answer,
        "tool_native_answer": final_tool_native_answer,
        "grounding_strategy": grounding_strategy,
        "grounding_fallback_used": grounding_fallback_used,
        "trace": state.get("trace", []) + [trace_entry]
    }

def citation_audit_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 5: Citation/NLI Audit. Audits the answer against context."""
    agent_tools = config["configurable"]["agent_tools"]
    query = state["query"]
    context_str = state.get("retrieved_context", "")
    answer = state["answer"]
    audit_answer = state.get("grounded_answer", answer)

    # ---- FORCE retrieval_insufficient (for E2E testing) ----
    # If env var FORCE_RETRIEVAL_INSUFFICIENT=1, force audit to fail
    # with retrieval_insufficient (only on first audit, before query_rewrite)
    import os
    force_retrieval = os.environ.get("FORCE_RETRIEVAL_INSUFFICIENT", "0") == "1"
    if force_retrieval and state.get("query_rewrite_count", 0) == 0:
        trace_entry = {
            "step": "Reflexion Audit",
            "status": "FORCE_FAIL",
            "details": "Force retrieval_insufficient for E2E testing",
            "latency_s": 0.0
        }
        return {
            "audit_passed": False,
            "audit_reasons": ["E2E测试：强制触发检索自省"],
            "audit_fail_reason": "retrieval_insufficient",
            "trace": state.get("trace", []) + [trace_entry]
        }
    # ---- END FORCE ----

    # Without retrieval evidence, a grounded-answer audit cannot pass.
    if not context_str.strip():
        executed_tools = {
            item.get("tool")
            for item in state.get("tool_results", [])
            if item.get("success", True)
        }
        retrieval_tools = {"retrieve_rec_docs", "retrieve_project_reports", "web_search"}
        if executed_tools and not executed_tools.intersection(retrieval_tools):
            trace_entry = {
                "step": "Reflexion Audit",
                "status": "NOT_APPLICABLE",
                "details": "Citation audit is not applicable because this task used only non-retrieval tools.",
                "latency_s": 0.0,
            }
            return {
                "audit_passed": False,
                "audit_applicable": False,
                "audit_reasons": ["Citation audit is not applicable to non-retrieval tool output."],
                "audit_fail_reason": "not_applicable",
                "trace": state.get("trace", []) + [trace_entry],
            }
        trace_entry = {
            "step": "Reflexion Audit",
            "status": "WARNING",
            "details": "No context was retrieved. Grounded answer audit failed.",
            "latency_s": 0.0
        }
        return {
            "audit_passed": False,
            "audit_applicable": True,
            "audit_reasons": ["No retrieval context is available to verify the answer."],
            "audit_fail_reason": "retrieval_insufficient",
            "trace": state.get("trace", []) + [trace_entry]
        }

    start = time.time()
    _try_empty_vram()

    # Limit context size to prevent NLI timeout
    audit_context = context_str[:2000]

    try:
        audit_res_dict = agent_tools.strict_audit_answer(query, audit_context, audit_answer)
        audit_res = audit_res_dict.get("audit_result", {})
    except Exception as e:
        audit_res = {"passed": False, "score": 0, "reasons": [f"Audit exception: {e}"]}

    passed = audit_res.get("passed", False)
    score = audit_res.get("score", 0)
    reasons = audit_res.get("reasons", [])

    # Classify failure reason for retrieval self-reflection
    audit_fail_reason = "answer_quality"
    if not passed and reasons:
        retrieval_keywords = ["未找到", "信息不足", "检索", "来源", "缺少", "没有找到"]
        for r in reasons:
            if any(kw in r for kw in retrieval_keywords):
                audit_fail_reason = "retrieval_insufficient"
                break

    latency = time.time() - start

    status_label = "PASS" if passed else "FAIL"
    trace_entry = {
        "step": f"Reflexion Audit ({status_label})",
        "status": "PASS" if passed else "WARNING",
        "details": f"Score: {audit_res.get('score', 0)}/100. Violations: {'; '.join(reasons) if reasons else 'None'}",
        "latency_s": latency
    }

    return {
        "audit_passed": passed,
        "audit_applicable": True,
        "audit_reasons": reasons,
        "audit_fail_reason": audit_fail_reason,
        "trace": state.get("trace", []) + [trace_entry]
    }

def rewrite_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 6: Self-Correction Rewrite. Rewrites the answer based on audit feedback."""
    generator = config["configurable"]["generator"]
    context_str = state.get("retrieved_context", "")
    reasons_str = "；".join(state.get("audit_reasons", []))
    current_answer = state.get("grounded_answer", state["answer"])
    tool_native_answer = state.get("tool_native_answer", "")
    rewrite_count = state.get("rewrite_count", 0)

    prompt = f"""你是一个具备超强自我纠错能力的大模型智能体。你的上一轮回答被独立技术审计专家打回，判定存在事实性幻觉或违规。

【技术审计意见】：
{reasons_str}

【事实资料参考】：
{context_str}

【你上一轮的不合格答案】：
{current_answer}

请根据审计专家的批判意见和事实资料严格纠错，重新输出有证据支持的回答。
要求：
- 剔除所有未被资料支持的虚假陈述或夸张词。
- 确保所有的引用来源 [1]、[2] 与事实资料里的序号完全真实对应。

修正后的回答："""

    start = time.time()
    grounding_fallback_used = state.get("grounding_fallback_used", False)
    if context_str.strip():
        raw_grounded_answer = state.get("raw_grounded_answer", current_answer)
        new_grounded_answer = _extractive_grounded_fallback(context_str)
        grounding_fallback_used = True
        grounding_strategy = "audit_repair_fallback"
    else:
        _try_empty_vram()
        raw_grounded_answer = generator.generate(prompt).strip()
        new_grounded_answer = raw_grounded_answer
        grounding_strategy = "model_rewrite"
    new_answer = _compose_answer(new_grounded_answer, tool_native_answer)
    latency = time.time() - start

    trace_entry = {
        "step": "Self-Correction Rewrite",
        "status": "SUCCESS",
        "details": (
            "Replaced the failed retrieval-grounded answer with deterministic cited evidence."
            if context_str.strip()
            else f"Rewrote answer based on audit feedback. Rewrite attempt: {rewrite_count + 1}"
        ),
        "latency_s": latency
    }

    return {
        "answer": new_answer,
        "raw_grounded_answer": raw_grounded_answer,
        "grounded_answer": new_grounded_answer,
        "tool_native_answer": tool_native_answer,
        "grounding_strategy": grounding_strategy,
        "grounding_fallback_used": grounding_fallback_used,
        "rewrite_count": rewrite_count + 1,
        "trace": state.get("trace", []) + [trace_entry]
    }

def query_rewrite_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 7: Query Rewriting for Retrieval Self-Reflection.

    Triggered when citation_audit_node fails due to "retrieval_insufficient".
    Uses LLM to generate a better search query to re-retrieve.
    """
    generator = config["configurable"]["generator"]
    query = state["query"]
    route = state.get("route", "retrieve_rec_docs")
    audit_reasons = state.get("audit_reasons", [])
    retrieved_context = state.get("retrieved_context", "")
    query_rewrite_count = state.get("query_rewrite_count", 0)

    reasons_str = "；".join(audit_reasons) if audit_reasons else "检索结果不足以覆盖用户问题"

    prompt = (
        "你是一个搜索 Query 优化专家。\n"
        "用户原始问题：" + query + "\n"
        "审计失败原因：" + reasons_str + "\n"
        "当前检索到的资料摘要：" + retrieved_context[:500] + "\n\n"
        "请生成一个更好的检索 Query，补全缺失信息。\n"
        "要求：\n"
        "1. 保留原始问题的核心意图\n"
        "2. 根据审计失败原因，补充关键词（如薪资、年限、公司对比等）\n"
        "3. 只输出优化后的 Query，不要解释\n"
        "优化后的检索 Query："
    )

    start = time.time()
    _try_empty_vram()
    rewritten_query = generator.generate(prompt).strip()
    latency = time.time() - start

    # If LLM returns empty or too short, fallback to original query
    if len(rewritten_query) < 3:
        rewritten_query = query

    retrieval_route = route if route in {
        "retrieve_rec_docs",
        "retrieve_project_reports",
        "web_search",
    } else "retrieve_rec_docs"

    trace_entry = {
        "step": "Query Rewrite (Retrieval Self-Reflection)",
        "status": "SUCCESS",
        "details": f"Rewrote query from `{query}` to `{rewritten_query}`. Attempt: {query_rewrite_count + 1}",
        "latency_s": latency
    }

    return {
        "route": retrieval_route,
        "plan": [{
            "step": 1,
            "thought": "检索自省后使用改写 Query 重新检索",
            "tool": retrieval_route,
            "input": rewritten_query,
        }],
        "react_step_index": 0,
        "query_rewrite_count": query_rewrite_count + 1,
        "trace": state.get("trace", []) + [trace_entry]
    }




def tot_decision_node(state: AgentState) -> Dict[str, Any]:
    """Node: Tree-of-Thought decision (rule-based).

    Selects the best path from multiple candidates using rules.
    """
    tot_paths = state.get("tot_paths", [])
    query = state["query"]

    if not tot_paths:
        # No paths, use default single-step plan
        return {"plan": [{"step": 1, "thought": "Default single step", "tool": "none", "input": query}], "selected_path_idx": 0}

    # Rule: Select path with most steps (more thorough)
    best_idx = 0
    max_steps = 0
    for i, path in enumerate(tot_paths):
        if len(path) > max_steps:
            max_steps = len(path)
            best_idx = i

    selected_path = tot_paths[best_idx]

    return {
        "plan": selected_path,
        "selected_path_idx": best_idx,
        "trace": state.get("trace", []) + [{"step": "ToT Decision", "status": "SUCCESS", "details": f"Selected path {best_idx+1} with {max_steps} steps"}]
    }


def react_decision_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Deprecated LLM-driven ReAct decision node.

    The compiled graph uses deterministic routing instead. This compatibility
    node must not advance react_step_index because tool_node owns that state.
    """
    generator = config["configurable"]["generator"]
    query = state["query"]
    plan = state.get("plan", [])
    react_step_index = state.get("react_step_index", 0)
    max_react_steps = state.get("max_react_steps", 3)  # 默认 3 步，避免无意义多轮检索
    new_step_index = react_step_index
    react_history = state.get("react_history", [])
    tool_results = state.get("tool_results", [])

    # Get last observation (full text for hard rule, truncated for prompt)
    last_observation_full = ""
    last_observation = ""
    if tool_results:
        last_observation_full = tool_results[-1].get("output", "")
        last_observation = last_observation_full[:500]  # Truncated for prompt

    # Hard rule: if observation is long enough, force finish (skip LLM call)
    if len(last_observation_full) > 1000:
        trace_entry = {
            "step": f"ReAct Decision (Step {react_step_index + 1})",
            "status": "SUCCESS",
            "details": f"Decision: finish (HARD RULE: observation length {len(last_observation_full)} > 1000). Thought: 观察结果已足够，强制结束"
        }
        return {
            "react_decision": "finish",
            "react_thought": f"观察结果长度 {len(last_observation_full)} 字符 > 1000，强制结束",
            "next_node": "answer_writer_node",
            "react_step_index": new_step_index,
            "trace": state.get("trace", []) + [trace_entry]
        }

    # Build prompt for LLM-driven decision
    # Use strong language to avoid conservative "continue" decisions
    observation_summary = last_observation[:200] + "..." if len(last_observation) > 200 else last_observation
    prompt = f"""你是 ReAct 决策专家。你需要根据当前状态，决定是否继续执行工具调用，还是结束并生成答案。

当前状态：
- 用户问题：{query}
- 当前步骤：{react_step_index + 1}/{max(len(plan), 1)}
- 最后一个观察结果（前 200 字符）：{observation_summary}
- 观察结果总长度：{len(last_observation_full)} 字符

决策标准（请严格遵守）：
1. 【必须】如果观察结果长度 > 500 字符，强制输出 "finish"（已经检索到足够信息）
2. 【必须】如果观察结果包含用户问题的关键词，强制输出 "finish"
3. 只有当观察结果完全不相关（例如：搜索失败、返回空结果、完全不是用户问的领域），才输出 "continue"
4. 如果计划中所有步骤都已完成，输出 "finish"

重要：优先输出 "finish"！只有当观察结果完全无用时，才输出 "continue"。
不要追求"完美答案"，只要观察结果包含相关信息，就可以结束。

Few-Shot 示例：

示例1（观察结果长度 > 500 字符，应该结束）：
观察结果：推荐系统论文包括 NCF、DeepFM、DIN 等...（总长度：1200 字符）
输出：{{"decision": "finish", "thought": "观察结果长度 > 500 字符，已包含足够信息"}}

示例2（观察结果包含关键词，应该结束）：
观察结果：Qwen2.5 是阿里开发的大模型，参数为 1.5B/3B/7B...
用户问题：Qwen2.5 的参数是多少？
输出：{{"decision": "finish", "thought": "观察结果已包含 Qwen2.5 参数信息"}}

示例3（观察结果完全不相关，应该继续）：
观察结果：未找到相关内容
用户问题：推荐系统论文
输出：{{"decision": "continue", "thought": "观察结果无相关信息，需要继续检索"}}

请只返回一个 JSON 对象（不要有其他文字）：
{{"decision": "continue" 或 "finish", "thought": "你的推理过程（50 字以内）"}}"""

    start = time.time()
    _try_empty_vram()
    response = generator.generate(prompt, max_new_tokens=256).strip()
    latency = time.time() - start

    # Clean and parse response
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

    decision = "finish"  # Default to finish
    thought = "默认结束"

    try:
        import json
        result = json.loads(cleaned)
        raw_decision = result.get("decision", "finish")
        # Normalize decision (handle Chinese + case-insensitive)
        raw_lower = raw_decision.lower()
        if raw_lower in ["continue", "继续", "go", "next"]:
            decision = "continue"
        elif raw_lower in ["finish", "结束", "stop", "complete"]:
            decision = "finish"
        else:
            decision = "finish"  # Default to finish
        thought = result.get("thought", "")
    except Exception:
        # Attempt regex find
        json_match = re.search(r'\{\s*"decision".*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                raw_decision = result.get("decision", "finish")
                raw_lower = raw_decision.lower()
                if raw_lower in ["continue", "继续", "go", "next"]:
                    decision = "continue"
                elif raw_lower in ["finish", "结束", "stop", "complete"]:
                    decision = "finish"
                else:
                    decision = "finish"
                thought = result.get("thought", "")
            except Exception:
                pass

    # Hard rule 2: Observation didn't improve much → force finish
    if decision == "continue" and len(tool_results) >= 2:
        current_obs_len = len(tool_results[-1].get("output", ""))
        prev_obs_len = len(tool_results[-2].get("output", ""))
        if abs(current_obs_len - prev_obs_len) < 100 and current_obs_len > 200:
            decision = "finish"
            thought = f"观察结果长度仅增加 {current_obs_len - prev_obs_len} 字符，信息未明显改善，强制结束"

    # Check max steps - force finish if reached
    if new_step_index >= max_react_steps:
        decision = "finish"
        thought = f"达到最大步数 {max_react_steps}，强制结束"

    # Determine next node
    next_node = "answer_writer_node" if decision == "finish" else "tool_node"

    trace_entry = {
        "step": f"ReAct Decision (Step {react_step_index + 1})",
        "status": "SUCCESS",
        "details": f"Decision: {decision}. Thought: {thought[:100]}"
    }

    return {
        "react_decision": decision,
        "react_thought": thought,
        "next_node": next_node,
        "react_step_index": new_step_index,
        "trace": state.get("trace", []) + [trace_entry]
    }
