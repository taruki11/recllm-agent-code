from __future__ import annotations
import time
import json
import re
from typing import Any, Dict, List, Tuple
from .schema import SearchResult
from .tools import AgentTools
from .qwen_generator import QwenGenerator
from .planner import Planner
from .memory import Memory

# Max chars of context fed into NLI audit — keeps forward pass under 30s on WDDM 3060
_AUDIT_CONTEXT_CHAR_LIMIT = 2000
# Max chars of observation fed back into history — prevents prompt explosion
_OBS_CHAR_LIMIT = 600

def _try_empty_vram() -> None:
    """Best-effort VRAM cleanup between heavy steps on WDDM GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

class AgentExecutor:
    """v27 Master Agent Executor orchestrating Multi-Step Planning, Memory Recall, and Reflexion Self-Audit.
    Aligns with DeepLearning.AI Pattern 3: Planning, Pattern 2: Tool Use, and Pattern 1: Reflection."""
    def __init__(self, registry: Any, tools: AgentTools, generator: QwenGenerator, memory: Memory | None = None) -> None:
        self.registry = registry
        self.tools = tools
        self.generator = generator
        self.planner = Planner(generator)
        self.memory = memory if memory is not None else Memory()

    def execute_react(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        trace_log = []
        start_time = time.time()

        # Retrieved evidence belongs to one query. Keep profile/history, but do
        # not let stale chunks leak into the next answer.
        self.memory.clear_context()

        # Recall Session Context & History
        history_summary = self.memory.get_history_summary()
        resume_profile = self.memory.get_user_profile("resume", "")
        
        trace_log.append({
            "step": "🕸️ Memory Recall",
            "status": "SUCCESS",
            "details": f"Recalled history turns: `{len(self.memory.history)}`. Cached resume status: `{'Active' if resume_profile else 'None'}`."
        })
        
        # Step 1: Multi-Step Task Planning
        print("[Planner] Decomposing query into sub-tasks...")
        steps_plan = self.planner.plan(query)
        plan_details = "<br>".join([f"Step {s.get('step', 1)}: {s.get('thought', '')} -> 🛠️ `{s.get('tool', 'none')}` with input `\"{s.get('input', '')}\"`" for s in steps_plan])
        
        trace_log.append({
            "step": "📅 Planner Deconstruct",
            "status": "SUCCESS",
            "details": f"Generated Multi-step plan ({len(steps_plan)} steps):<br>{plan_details}"
        })
        
        context_accumulated: List[SearchResult] = []
        observation_history = ""
        
        # Step 2: Sequential Execution of planned steps
        for step_info in steps_plan:
            step_num = step_info.get("step", 1)
            thought = step_info.get("thought", "正在执行计划中的步骤...")
            tool = step_info.get("tool", "none")
            tool_input = step_info.get("input", query)
            
            if tool == "none":
                continue
                
            trace_log.append({
                "step": f"🛠️ Step {step_num} - Thought & Action",
                "status": "SUCCESS",
                "details": f"**Thought**: {thought}<br>**Executing Tool**: `{tool}` with input `\"{tool_input}\"`"
            })
            
            observation = "没有获取到有效观察结果。"
            
            # Smart Memory Matching: If compare_resume_jd is executed, dynamically retrieve resume from user profile memory
            if tool == "compare_resume_jd":
                cached_resume = self.memory.get_user_profile("resume")
                if cached_resume:
                    trace_log.append({
                        "step": "🧠 Memory Injection",
                        "status": "SUCCESS",
                        "details": "Successfully injected cached user resume from Session Memory into compare_resume_jd argument."
                    })
                    # Format comparison argument combining cached resume and JD input
                    comparison_text = f"【候选人简历】：\n{cached_resume}\n\n【匹配岗位JD描述】：\n{tool_input}"
                    tool_res = self.tools.compare_resume_jd(comparison_text, "推荐系统算法工程师岗")
                    observation = tool_res.get("comparison", "")
                else:
                    tool_res = self.tools.compare_resume_jd(tool_input, "推荐系统算法工程师岗")
                    observation = tool_res.get("comparison", "")
                    
            elif tool in ["retrieve_rec_docs", "retrieve_project_reports"]:
                tool_info = self.registry.get_tool(tool)
                if tool_info:
                    tool_func = tool_info["func"]
                    tool_res = tool_func(self.tools, tool_input)
                    
                    chunks_dicts = tool_res.get("results", [])
                    results_text = []
                    temp_results = []
                    
                    from .schema import Chunk, SearchResult
                    for idx, cd in enumerate(chunks_dicts, 1):
                        title = cd["chunk"]["title"]
                        text = cd["chunk"]["text"]
                        results_text.append(f"[{idx}] {title}: {text[:350]}")
                        
                        # Reconstruct search results
                        ch = Chunk.from_dict(cd["chunk"])
                        sr = SearchResult(chunk=ch, score=cd.get("score", 1.0), rank=idx)
                        temp_results.append(sr)
                        context_accumulated.append(sr)
                        
                    # Cache context chunks in memory
                    self.memory.cache_context(temp_results)
                    observation = "\n\n".join(results_text) if results_text else "检索库中没有找到相关匹配条目。"
                    
            elif tool == "generate_mock_questions":
                tool_res = self.tools.generate_mock_questions(tool_input)
                observation = tool_res.get("mock_questions", "")
            else:
                observation = f"未知工具名 '{tool}'。"
                
            # Free VRAM between tool steps to reduce fragmentation on WDDM GPU
            _try_empty_vram()

            trace_log.append({
                "step": f"📦 Step {step_num} - Observation",
                "status": "SUCCESS",
                "details": f"**Observation Result (Truncated)**:<br>{observation[:250]}..."
            })

            observation_history += f"\n[Step {step_num} Tool: {tool} Observation]:\n{observation[:_OBS_CHAR_LIMIT]}\n\n"
            
        # Step 3: Synthesis and Draft Formulation
        # Retrieve all cached context str
        context_str = self.memory.get_cached_context_str(limit=6)
        
        draft_prompt = f"""你是一个高级的推荐系统垂直智能体专家。请综合用户的提问、我们多步规划执行所得到的参考事实，以及会话的历史记忆，给出最为专业、客观且严谨的终极回答。

【会话历史问答概要】：
{history_summary}

【多步工具调用所得事实资料】：
{context_str}

【多步工具调用所得观察记录】：
{observation_history}

【当前用户的问题】：
{query}

要求：
1. 必须完全基于上述事实资料和观察记录进行分析，切勿编造虚假信息。
2. 每一个关键论点或事实结论，必须使用 [1] [2] 标注其在“多步工具调用所得事实资料”中的序号来源。
3. 严禁使用任何夸大、绝对化的表达（如“完美解决”、“100%成功”）。

请开始回答："""
        
        print("[Writer] Compiling draft answer...")
        draft_answer = self.generator.generate(draft_prompt).strip()
        
        # Step 4: Reflexion Self-Audit & Correction Loop
        current_answer = draft_answer
        attempt = 1
        max_rewrites = 3
        passed = False
        reasons: List[str] = []
        audit_fail_reason = "answer_quality"

        while attempt <= max_rewrites:
            # Without retrieval evidence, a grounded-answer audit cannot pass.
            if not context_str.strip():
                passed = False
                reasons = ["No retrieval context is available to verify the answer."]
                audit_fail_reason = "retrieval_insufficient"
                trace_log.append({
                    "step": "🛡️ Reflexion Audit - FAIL",
                    "status": "WARNING",
                    "details": "No context was retrieved, so the answer cannot be verified."
                })
                break

            # Truncate context before audit to prevent NLI timeout on WDDM GPU
            audit_context = context_str[:_AUDIT_CONTEXT_CHAR_LIMIT]

            print(f"[Critic] Auditing answer draft (Attempt {attempt}, ctx={len(audit_context)} chars)...")
            _try_empty_vram()
            try:
                audit_res_dict = self.tools.strict_audit_answer(query, audit_context, current_answer)
                audit_res = audit_res_dict.get("audit_result", {})
            except Exception as audit_err:
                print(f"[Critic] Audit exception: {audit_err}")
                audit_res = {"passed": False, "score": 0, "reasons": [f"Audit exception: {audit_err}"]}
            passed = audit_res.get("passed", False)
            reasons = audit_res.get("reasons", [])
            
            if passed:
                trace_log.append({
                    "step": f"🛡️ Reflexion Audit - Pass (Attempt {attempt})",
                    "status": "PASS",
                    "details": f"⭐ *通过了 NLI 事实一致性与绝对化表述审计评估！*<br>**Score**: `{audit_res.get('score', 100)}/100`"
                })
                break
            else:
                reasons_str = "；".join(reasons)
                trace_log.append({
                    "step": f"⚠️ Reflexion Audit - FAIL (Attempt {attempt})",
                    "status": "WARNING",
                    "details": f"**Audit Score**: `{audit_res.get('score', 0)}/100`<br>**Violations**: {reasons_str}<br>🔁 *触发自我修正反思，重新构建优化后的回答...*"
                })
                
                # Format critique prompt for reflexion correction
                critique_prompt = f"""你是一个具备超强自我纠错能力的大模型智能体。你的上一轮回答被独立技术审计专家打回，判定存在事实性幻觉或违规。

【技术审计意见】：
{reasons_str}

【事实资料参考】：
{context_str}

【你上一轮的不合格答案】：
{current_answer}

请根据审计专家的批判意见和事实资料，严格纠错，重新输出完美校准后的回答。
要求：
- 剔除所有未被资料支持的虚假陈述或夸张词。
- 确保所有的引用来源 [1]、[2] 与事实资料里的序号完全真实对应。

修正后的回答："""
                
                attempt += 1
                current_answer = self.generator.generate(critique_prompt).strip()
                
        final_answer = current_answer
        
        # Save Q&A turn into conversation history memory
        self.memory.add_history(query, final_answer)
        
        total_latency = time.time() - start_time
        
        return {
            "query": query,
            "answer": final_answer,
            "trace_log": trace_log,
            "context_accumulated": context_accumulated,
            "audit_passed": passed,
            "audit_reasons": reasons,
            "audit_fail_reason": "" if passed else audit_fail_reason,
            "total_latency_s": total_latency
        }
