# -*- coding: utf-8 -*-
"""
recllm_langgraph/tools.py
=========================
LangChain Tool 封装层 — 将 RecLLM-Agent 原生工具注册为 @tool 可调用对象。

为什么需要这层封装？
--------------------
RecLLM-Agent 的原生 AgentTools 是自研 Python 类，调用方式为：
    agent_tools.retrieve_rec_docs(query)  →  {"results": [...]}

LangChain 生态（LangGraph / LangSmith / 工具调用链）需要符合 @tool 协议的
可序列化函数对象，才能支持：
  - LangSmith 自动追踪工具调用输入/输出
  - 未来接入 OpenAI Function Calling / Anthropic Tool Use 格式
  - ToolNode 的自动分发（根据工具名路由）

封装策略：闭包 + @tool 装饰器
    get_langchain_tools(agent_tools) 函数通过闭包捕获 agent_tools 实例，
    在内部定义 @tool 装饰的函数，实现零侵入式包装。
    原始 AgentTools 完全不受影响。

工具列表
--------
1. retrieve_rec_docs        — 检索推荐系统学术文献/算法概念
2. retrieve_project_reports — 检索工程项目报告/简历/需求文档
3. compare_resume_jd        — 简历 vs JD 深度对比分析
4. generate_mock_questions  — 推荐/搜索领域面试题生成
"""

from typing import Any, Dict, List

from langchain_core.tools import tool

from recllm_agent.tools import AgentTools


def get_langchain_tools(agent_tools: AgentTools) -> List[Any]:
    """
    将 RecLLM-Agent 原生工具包装为 LangChain ``@tool`` 对象列表。

    通过闭包捕获 ``agent_tools`` 实例，避免全局状态污染。
    返回的工具列表可直接传入：
      - ``ToolNode``（LangGraph）
      - ``create_react_agent``（LangChain）
      - ``AgentExecutor``（LangChain）

    Parameters
    ----------
    agent_tools : AgentTools
        已初始化的 AgentTools 实例（含 BM25/Dense 索引、Qwen 生成器）。

    Returns
    -------
    List[Any]
        4 个 @tool 装饰的可调用对象列表。
    """

    @tool("retrieve_rec_docs")
    def retrieve_rec_docs(query: str) -> Dict[str, Any]:
        """
        检索推荐系统领域学术文献与算法概念知识库。

        适用场景：
          - 推荐算法原理（DIN, SASRec, BERT4Rec, DSSM, LightGCN...）
          - LLM4Rec 相关论文与方法（P5, TALLRec, RecInterpreter...）
          - Transformer 结构、注意力机制、序列推荐等技术概念

        Parameters
        ----------
        query : str
            用户查询文本（中英文均支持）。

        Returns
        -------
        Dict[str, Any]
            {"results": List[SearchResult], "total": int}
        """
        return agent_tools.retrieve_rec_docs(query)

    @tool("retrieve_project_reports")
    def retrieve_project_reports(query: str) -> Dict[str, Any]:
        """
        检索本项目工程报告、评估日志、简历及系统需求文档。

        适用场景：
          - 项目指标查询（Recall@K, NDCG@K, 引用覆盖率等）
          - SFT/DPO 训练配置与结果
          - Reflexion 审计日志分析
          - 候选人简历内容提取（用于 JD 匹配）

        Parameters
        ----------
        query : str
            查询文本，如"v18 SFT 的引用覆盖率是多少"。

        Returns
        -------
        Dict[str, Any]
            {"results": List[SearchResult], "total": int}
        """
        return agent_tools.retrieve_project_reports(query)

    @tool("compare_resume_jd")
    def compare_resume_jd(resume_jd_text: str) -> Dict[str, Any]:
        """
        深度对比候选人简历与目标岗位 JD，输出匹配度分析报告。

        输入格式说明：
          - 若 Memory 中已缓存简历，可直接传入 JD 文本；
          - 否则传入"【候选人简历】：...\\n\\n【匹配岗位JD描述】：..."格式。

        输出包含：
          - 技能匹配度评分
          - 优势与差距分析
          - 面试准备建议

        Parameters
        ----------
        resume_jd_text : str
            JD 文本 或 简历+JD 拼接文本。

        Returns
        -------
        Dict[str, Any]
            {"comparison": str}  —— 结构化对比报告文本
        """
        return agent_tools.compare_resume_jd(resume_jd_text, "推荐系统算法工程师岗")

    @tool("generate_mock_questions")
    def generate_mock_questions(topic: str) -> Dict[str, Any]:
        """
        生成指定推荐系统/搜索主题的深度面试题及考察要点。

        适用场景：
          - 推荐算法原理（协同过滤、矩阵分解、神经网络推荐）
          - LLM 对齐（SFT, RLHF, DPO, PPO）
          - 检索工程（BM25, Dense Retrieval, ANN, Faiss, HNSW）
          - 向量数据库与 ANN 搜索（Milvus, Weaviate, Pinecone）
          - RAG 架构设计与评估

        Parameters
        ----------
        topic : str
            面试题主题，如"ANN 搜索与 Faiss/HNSW 算法原理"。

        Returns
        -------
        Dict[str, Any]
            {"mock_questions": str}  —— 含题目、考察点、参考答案的 Markdown 文本
        """
        return agent_tools.generate_mock_questions(topic)

    @tool("web_search")
    def web_search(query: str) -> Dict[str, Any]:
        """
        使用 DuckDuckGo 免费接口联网搜索（无需 API key）。

        适用场景：
          - 实时信息（最新论文、新闻、技术动态）
          - 本地知识库未覆盖的内容
          - 需要验证的事实性信息

        Parameters
        ----------
        query : str
            搜索查询文本。

        Returns
        -------
        Dict[str, Any]
            {"results": List[Dict], "total": int}  —— 搜索结果列表
        """
        return agent_tools.web_search(query)

    return [
        retrieve_rec_docs,
        retrieve_project_reports,
        compare_resume_jd,
        generate_mock_questions,
        web_search,
    ]
