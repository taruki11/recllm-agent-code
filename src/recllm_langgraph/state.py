# -*- coding: utf-8 -*-
"""
recllm_langgraph/state.py
=========================
LangGraph AgentState — v28 RecLLM-Agent 工作流的集中状态容器。

所有节点（planner / router / tool / writer / auditor / rewriter）
共享同一个 AgentState 实例。LangGraph 在每次节点返回 Dict 后，
自动将返回值 **merge** 到当前 State 中（类似 Redux reducer）。

State 流转示意：
    START
      │ {query}
      ▼
    planner_node     → 追加 {plan, trace}
      │
      ▼
    router_node      → 追加 {route, trace}
      │
      ▼
    tool_node        → 追加 {tool_results, retrieved_context, trace}
      │
      ▼
    answer_writer_node → 追加 {answer, trace}
      │
      ▼
    citation_audit_node → 追加 {audit_passed, audit_reasons, audit_fail_reason, trace}
      │
      ├──[PASS]──────────────────────────────────> END
      └──[FAIL]──> rewrite_node → 追加 {answer, rewrite_count, trace}
                           └──────────────> citation_audit_node (循环)
      └──[FAIL + retrieval_insufficient]──> query_rewrite_node → 追加 {query, query_rewrite_count, trace}
                                                           └──> tool_node (重新检索)
"""

from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    LangGraph 工作流的全局执行状态。

    Attributes
    ----------
    query : str
        用户原始输入问题。
    plan : List[Dict[str, Any]]
        Planner 输出的多步执行计划。每个 step 格式：
        {"step": int, "thought": str, "tool": str, "input": str}
    route : str
        IntentRouter 分类的工具名称。
        可选值：retrieve_rec_docs | retrieve_project_reports |
                compare_resume_jd | generate_mock_questions | web_search | none
    tool_results : List[Dict[str, Any]]
        工具调用的原始输出列表。每项格式：
        {"tool": str, "input": str, "output": str}
    retrieved_context : str
        RAG 检索到的上下文文本（多个 Chunk 拼接），供 LLM 生成时使用。
    answer : str
        当前答案草稿（由 answer_writer_node 生成或由 rewrite_node 修正）。
    audit_passed : bool
        NLI 引用审计结果。True = 通过，False = 存在幻觉或违规。
    audit_applicable : bool
        当前任务是否适合使用引用审计。非检索工具原生输出为 False。
    audit_reasons : List[str]
        审计未通过时的违规原因列表（e.g., "URL hallucination detected"）。
    audit_fail_reason : str
        审计失败原因分类："answer_quality"（答案写得不好）或
        "retrieval_insufficient"（检索结果不足）。
        用于决定下一步是 rewrite_node 还是 query_rewrite_node。
    rewrite_count : int
        Reflexion 重写循环已触发次数。达到上限（2次）后强制结束，
        避免死循环消耗 VRAM。
    query_rewrite_count : int
        检索改写循环已触发次数。达到上限（2次）后强制结束，
        避免死循环。
    max_rewrite_attempts : int
        回答重写上限。默认 1 次。
    max_query_rewrites : int
        检索查询改写上限。默认 1 次。
    trace : List[Dict[str, Any]]
        完整节点执行轨迹日志（可观测性字段）。每个节点追加一条：
        {"step": str, "status": str, "details": str, "latency_s": float}
    latency_s : float
        整个 E2E 会话总耗时（秒），由调用方统计写入。
    # ---- ReAct 相关字段（v29 新增）----
    react_step_index : int
        当前 ReAct 循环执行到第几步（0-based）。
    react_history : List[Dict[str, Any]]
        ReAct 执行历史，每步记录：
        {"thought": str, "action": str, "action_input": str, "observation": str}
    max_react_steps : int
        最大 ReAct 循环步数（默认 3 步），防止无限循环。
    """

    query: str
    plan: List[Dict[str, Any]]
    route: str
    tool_results: List[Dict[str, Any]]
    retrieved_context: str
    answer: str
    raw_grounded_answer: str
    grounded_answer: str
    tool_native_answer: str
    grounding_strategy: str
    grounding_fallback_used: bool
    audit_passed: bool
    audit_applicable: bool
    audit_reasons: List[str]
    audit_fail_reason: str
    rewrite_count: int
    query_rewrite_count: int
    max_rewrite_attempts: int
    max_query_rewrites: int
    trace: List[Dict[str, Any]]
    latency_s: float
    # ---- ReAct 新增字段 ----
    react_step_index: int
    react_history: List[Dict[str, Any]]
    max_react_steps: int

    # ---- ReAct 新增字段 (v30) ----
    react_decision: str  # LLM 决策结果："continue" 或 "finish"
    react_thought: str  # LLM 决策的推理过程
    next_node: str  # LLM 决策后的下一个节点

    # ---- Tree-of-Thought (ToT) 新增字段 (v29) ----
    tot_paths: List[List[Dict[str, Any]]]  # 多条规划路径
    selected_path_idx: int  # 选中的路径索引
    tot_explored_count: int  # 已探索的路径数
