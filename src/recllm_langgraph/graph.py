# -*- coding: utf-8 -*-
"""
recllm_langgraph/graph.py
=========================
LangGraph StateGraph 工作流定义 — v29 RecLLM-Agent（含 ReAct 规则循环）。

工作流拓扑（有向循环图）：
    START
      │
      ▼
    planner_node        # 多步任务规划
      │
      ▼
    router_node         # 意图分类 → 工具选择
      │
      ▼
    tool_node           # 单步工具执行（ReAct 风格）
      │
      ▼
    should_continue_react()  # 规则判断：继续 or 结束
      ├──[tool_node]─────────> tool_node  # 执行下一步
      │
      ▼
    answer_writer_node  # Qwen2.5-3B LLM 答案生成
      │
      ▼
    citation_audit_node # 规则预检 + LLM judge 式引用/事实审计
      │
      ├──[PASS]────────────────────────> END
      │
      ├──[FAIL + answer_quality + rewrite_count < 2]──> rewrite_node
      │              └──> citation_audit_node  # 重新审计（循环）
      │
      └──[FAIL + retrieval_insufficient + query_rewrite_count < 2]──> query_rewrite_node
                                 └──> tool_node  # 重新检索
"""

from langgraph.graph import StateGraph, START, END
from recllm_langgraph.state import AgentState
from recllm_langgraph.nodes import (
    planner_node,
    router_node,
    tool_node,
    answer_writer_node,
    citation_audit_node,
    rewrite_node,
    query_rewrite_node,
)


def route_after_audit(state: AgentState) -> str:
    """
    citation_audit_node 之后的条件路由函数。

    Logic
    -----
    - 若审计通过（audit_passed=True），终止工作流。
    - 若审计失败原因 = "retrieval_insufficient"：
        - 若已达最大改写次数，强制终止。
        - 否则，路由到 query_rewrite_node 触发检索自省。
    - 若审计失败原因 = "answer_quality"（或默认）：
        - 若已达最大重写次数，强制终止。
        - 否则，路由到 rewrite_node 触发 Reflexion 自纠错。

    Parameters
    ----------
    state : AgentState
        当前节点执行后的全局状态。

    Returns
    -------
    str
        目标节点名称（END 表示终止）。
    """
    passed = state.get("audit_passed", False)
    if state.get("audit_applicable") is False:
        return END

    # 终止条件：审计通过
    if passed:
        return END

    audit_fail_reason = state.get("audit_fail_reason", "answer_quality")
    rewrite_count = state.get("rewrite_count", 0)
    query_rewrite_count = state.get("query_rewrite_count", 0)
    max_rewrite_attempts = max(0, state.get("max_rewrite_attempts", 1))
    max_query_rewrites = max(0, state.get("max_query_rewrites", 1))

    # 检索自省分支：改写 Query 重新检索
    if audit_fail_reason == "retrieval_insufficient":
        if query_rewrite_count >= max_query_rewrites:
            return END  # 强制终止，避免死循环
        return "query_rewrite_node"

    # 回答自省分支：重写回答（默认）
    if rewrite_count >= max_rewrite_attempts:
        return END  # 强制终止
    return "rewrite_node"


def should_continue_react(state: AgentState) -> str:
    """
    ReAct 循环的条件路由函数（规则版：不用 LLM）。
    
    逻辑：
    - 若已完成步骤数 >= 总步骤数，进入 answer_writer_node
    - 若达到最大 ReAct 步数，强制进入 answer_writer_node
    - 否则，返回 tool_node 执行下一步
    
    Parameters
    ----------
    state : AgentState
        当前节点执行后的全局状态。
    
    Returns
    -------
    str
        目标节点名称（"tool_node" 或 "answer_writer_node"）。
    """
    react_step_index = state.get("react_step_index", 0)
    max_react_steps = state.get("max_react_steps", 3)  # 默认 3 步
    plan = state.get("plan", [])
    
    # 强制终止条件：达到最大步数
    if react_step_index >= max_react_steps:
        return "answer_writer_node"
    
    # 所有计划步骤已完成
    if react_step_index >= len(plan):
        return "answer_writer_node"
    
    # 默认：继续执行下一步
    return "tool_node"


def compile_workflow():
    """
    构建并编译 RecLLM-Agent StateGraph 工作流（含 ReAct 规则循环）。
    
    Returns
    -------
    CompiledGraph
        LangGraph 编译后的可执行工作流对象。
        调用方式：graph.invoke({"query": "..."}, config={"configurable": {...}})
    
    Notes
    -----
    config["configurable"] 中需注入以下组件（避免重复加载 VRAM）：
        - "generator": QwenGenerator 实例（RTX 3060 12GB 单实例共享）
        - "planner":   Planner 实例
        - "router":    IntentRouter 实例
        - "agent_tools": AgentTools 实例
        - "memory":    AgentMemory 实例
    
    Example
    -------
    >>> graph = compile_workflow()
    >>> result = graph.invoke(
    ...     {"query": "SASRec 和 BERT4Rec 的区别是什么？"},
    ...     config={"configurable": {
    ...         "generator": qwen_gen,
    ...         "planner": planner,
    ...         "router": router,
    ...         "agent_tools": tools,
    ...         "memory": memory,
    ...     }}
    ... )
    >>> print(result["answer"])
    """
    workflow = StateGraph(AgentState)

    # ── 节点注册 ──────────────────────────────────────────────────────────
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("router_node", router_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("answer_writer_node", answer_writer_node)
    workflow.add_node("citation_audit_node", citation_audit_node)
    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("query_rewrite_node", query_rewrite_node)  # 新增：检索自省

    # ── 线性边（主流程）───────────────────────────────────────────────────
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "router_node")
    workflow.add_edge("router_node", "tool_node")
    # tool_node 之后不直接到 answer_writer_node，而是先判断是否继续 ReAct 循环
    workflow.add_edge("answer_writer_node", "citation_audit_node")

    # ── ReAct deterministic loop ───────────────────────────────────
    # The tool node owns react_step_index. Route from that index so each
    # planned step is executed exactly once and in order.
    workflow.add_conditional_edges(
        "tool_node",
        should_continue_react,
        {
            "tool_node": "tool_node",  # 继续执行下一步
            "answer_writer_node": "answer_writer_node",  # 所有步骤完成，进入答案生成
        }
    )

    # ── 条件边（Reflexion Loop + 检索自省）─────────────────────────────
    # citation_audit_node → END  (审计通过 or 达到重写/改写上限)
    # citation_audit_node → rewrite_node  (审计失败 + answer_quality)
    # citation_audit_node → query_rewrite_node  (审计失败 + retrieval_insufficient)
    workflow.add_conditional_edges(
        "citation_audit_node",
        route_after_audit,
        {
            END: END,
            "rewrite_node": "rewrite_node",
            "query_rewrite_node": "query_rewrite_node",  # 新增
        },
    )

    # rewrite_node 修正后重新进入审计节点（循环）
    workflow.add_edge("rewrite_node", "citation_audit_node")

    # query_rewrite_node 改写 Query 后重新检索（检索自省循环）
    workflow.add_edge("query_rewrite_node", "tool_node")

    return workflow.compile()
