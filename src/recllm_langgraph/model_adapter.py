# -*- coding: utf-8 -*-
"""
recllm_langgraph/model_adapter.py
==================================
QwenChatAdapter — 本地 QwenGenerator 接入 LangChain 生态的桥接适配器。

设计背景
--------
LangChain / LangGraph 的标准 LLM 接口为 `BaseChatModel`（或 `BaseLLM`）。
我们的 `QwenGenerator` 是一个独立的本地推理封装，不继承 LangChain 基类。
直接使用时无法支持：
  - LangSmith Tracing（调用链可观测）
  - LCEL（LangChain Expression Language）管道组合
  - LangGraph node 的标准 `invoke()` 接口

解决方案：Adapter Pattern
  QwenChatAdapter(BaseChatModel)
    └── .generator = QwenGenerator  （委托调用）

这样做的好处：
  1. 核心算法（QwenGenerator）保持纯 Python，不依赖 LangChain 版本。
  2. Adapter 层极薄（< 30 行实现代码），升级框架版本时只改 Adapter。
  3. 单实例共享：所有节点通过 config["configurable"]["generator"] 共用
     同一个 QwenGenerator，不重复加载 3B 权重（3060 12GB VRAM 限制）。

LangSmith Tracing（v29 扩展预留）
  设置环境变量后自动生效：
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your_key>
  所有经过此 Adapter 的调用将自动上传 Trace。
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from recllm_agent.qwen_generator import QwenGenerator


class QwenChatAdapter(BaseChatModel):
    """
    将本地离线 ``QwenGenerator`` 包装为标准 LangChain ``BaseChatModel``。

    支持功能：
      - LangChain LCEL 管道（``|`` 运算符）
      - LangSmith Tracing（环境变量开启后自动生效）
      - LangGraph 节点内 ``invoke()`` 调用
      - 多轮消息拼接（message list → prompt string）

    Parameters
    ----------
    generator : QwenGenerator
        已加载到 GPU 的 QwenGenerator 实例。
        由外部统一初始化并通过 config["configurable"] 注入，
        确保整个工作流期间只有一份 3B 模型权重在 VRAM 中。

    Example
    -------
    >>> adapter = QwenChatAdapter(generator=qwen_gen)
    >>> response = adapter.invoke("SASRec 和 BERT4Rec 的核心区别？")
    >>> print(response.content)
    """

    # Pydantic v2 配置：允许非 Pydantic 类型的字段（QwenGenerator）
    model_config = {"arbitrary_types_allowed": True}

    generator: QwenGenerator = Field(
        description="已加载到 GPU 的 QwenGenerator 实例，所有节点共享单实例。"
    )

    @property
    def _llm_type(self) -> str:
        """LangChain 内部标识符，用于日志和 LangSmith Trace 标注。"""
        return "qwen-local-adapter"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        执行推理，将 LangChain 消息列表转换为 Qwen 输入 prompt。

        Parameters
        ----------
        messages : List[BaseMessage]
            LangChain 标准消息列表（HumanMessage / SystemMessage / AIMessage）。
        stop : Optional[List[str]]
            停止词列表（此 Adapter 传递给 QwenGenerator 但当前版本忽略）。
        run_manager : Optional[Any]
            LangChain 回调管理器（LangSmith Tracing 钩子在此触发）。

        Returns
        -------
        ChatResult
            包含 ``AIMessage`` 的 ``ChatResult`` 对象。
        """
        # 将多轮消息拼接为单 prompt（Qwen 本地推理格式）
        prompt = "\n".join(msg.content for msg in messages)

        # 委托 QwenGenerator 执行本地 GPU 推理
        response_text = self.generator.generate(prompt)

        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型标识参数，用于 LangSmith Trace 记录模型元信息。"""
        model_path = (
            self.generator.model_path
            if hasattr(self.generator, "model_path")
            else "unknown"
        )
        return {"model_path": model_path, "adapter_version": "v28"}
