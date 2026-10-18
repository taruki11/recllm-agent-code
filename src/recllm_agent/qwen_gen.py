from __future__ import annotations

from .schema import SearchResult


ACRONYM_GUARD = """固定术语和边界：
- RAG = Retrieval-Augmented Generation，中文可写“检索增强生成”。
- SFT = Supervised Fine-Tuning，中文可写“监督微调”。
- LoRA = Low-Rank Adaptation，中文可写“低秩适配”。
- DPO = Direct Preference Optimization，中文可写“直接偏好优化”。
- RLHF = Reinforcement Learning from Human Feedback，中文可写“基于人类反馈的强化学习”。
- reranker 是“重排器/重排模型”，不是召回器。
- DPO 不单独训练 reward model，而是用 chosen/rejected 偏好对直接优化 policy；RLHF 通常包含 reward model 和 PPO 等强化学习优化步骤。
- RTX 3060 12GB 适合 RAG、Agent、embedding 检索、reranker、Qwen2.5-1.5B LoRA/QLoRA 和小规模 DPO；不建议做 Qwen 7B LoRA、PPO、GRPO、多模态大模型训练或从零训练大模型。
- PPO/GRPO 如果检索资料没有给出全称，就只写缩写，不要自行展开。
如果检索资料没有要求展开缩写，优先只写缩写；不要把这些缩写解释成其他含义。资料列出核心模块清单时要完整列出，不要用“等”省略关键项。
"""


QWEN_CITATION_PROMPT_TEMPLATE = """你是一个推荐系统领域的学习助手。请根据以下检索到的资料回答问题。

要求：
1. 只使用下面提供的资料回答，不要编造信息。
2. 每条关键信息后面标注来源编号，如 [1]、[2]。
3. 如果资料不足以完整回答问题，请明确说"根据现有资料无法完全回答"。
4. 回答要准确、简洁，不要过度声称。

{acronym_guard}

---

检索资料：

{context}

---

问题：{question}

回答："""


QWEN_BASELINE_PROMPT_TEMPLATE = """以下是根据检索资料生成的 baseline 回答模板：

{extractive_answer}

请把上面的 baseline 回答改成更自然、流畅的中文段落，同时保留所有引用编号如 [1]、[2]。
不要添加 baseline 中不存在的信息，不要编造。"""


def build_citation_prompt(question: str, results: list[SearchResult], max_chars: int = 3000) -> str:
    blocks = []
    total_chars = 0
    for result in results:
        text = result.chunk.text.replace("\n", " ").strip()
        blocks.append(f"[{result.rank}] {result.chunk.title}: {text}")
        total_chars += len(text)
        if total_chars > max_chars:
            break
    context = "\n\n".join(blocks)
    return build_citation_prompt_from_context(question, context)


def build_citation_prompt_from_context(question: str, context: str) -> str:
    return QWEN_CITATION_PROMPT_TEMPLATE.format(
        acronym_guard=ACRONYM_GUARD,
        context=context,
        question=question,
    )


def build_baseline_upgrade_prompt(question: str, extractive_answer: str) -> str:
    return QWEN_BASELINE_PROMPT_TEMPLATE.format(
        extractive_answer=extractive_answer, question=question
    )
