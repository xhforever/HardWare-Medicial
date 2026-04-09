"""Regression tests for token budgeting, summary maintenance, and cache hits."""

from copy import deepcopy
from html import escape
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.executor import (  # noqa: E402
    _build_personalization_guidance,
    _department_prompt_label,
    _domain_display_name,
    _extract_personalization_preferences,
    _sanitize_prompt_payload,
    build_executor_plan,
)
from app.agents.judge_need_rag import JudgeNeedRAGAgent  # noqa: E402
from app.agents.memory import MemoryReadAgent  # noqa: E402
from app.core.config import EXECUTOR_PROMPT_MAX_TOKENS  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.services.cache_service import cache_service  # noqa: E402
from app.services.token_budget_service import estimate_tokens  # noqa: E402


def _make_long_text(seed: str, repeat: int = 24) -> str:
    return " ".join([seed] * repeat)


def _legacy_recent_history_text(state) -> str:
    lines = []
    for item in state.get("conversation_history", [])[-5:]:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}: {item.get('content', '')}")
    return "\n".join(lines)


def _legacy_rag_context_text(state) -> str:
    chunks = []
    for chunk in (state.get("rag_context") or [])[:5]:
        content = str(chunk.get("content", "")).strip()
        if content:
            chunks.append(content)
    return "\n\n".join(chunks) or "No retrieved context."


def _legacy_executor_prompt(state) -> str:
    question = state["question"]
    domain = state.get("domain", "general")
    memory_context = state.get("memory_context") or "No persistent memory context."
    ecg_info = state.get("ecg_metrics", "").strip() or "暂无最新数据"
    user_preferences = _extract_personalization_preferences(state)
    personalization_guidance = _build_personalization_guidance(user_preferences)

    sanitized_question = _sanitize_prompt_payload(question, fallback="未提供用户问题。")
    sanitized_memory_context = _sanitize_prompt_payload(
        memory_context,
        fallback="暂无长期画像。",
    )
    sanitized_history_text = _sanitize_prompt_payload(
        _legacy_recent_history_text(state) or "暂无历史对话",
        fallback="暂无历史对话。",
    )
    sanitized_rag_text = _sanitize_prompt_payload(
        _legacy_rag_context_text(state),
        fallback="暂无检索资料。",
    )
    sanitized_web_evidence = _sanitize_prompt_payload(
        "暂无联网资料",
        fallback="暂无联网资料。",
    )
    sanitized_ecg_info = _sanitize_prompt_payload(ecg_info, fallback="暂无最新数据。")
    sanitized_guidance = escape(personalization_guidance, quote=False)
    runtime_domain = escape(_domain_display_name(domain), quote=False)
    runtime_department = escape(
        _department_prompt_label(state.get("primary_department")),
        quote=False,
    )

    return (
        "<system_instructions>\n"
        "你是一位有温度、谨慎且专业的中文个人医疗助手。\n"
        "输出必须使用简体中文（必要的医学名词可保留英文缩写）。\n"
        "不要过度诊断；证据不足时明确说明不确定性。\n"
        "回答格式必须遵循：\n"
        "1) 先直接回应用户当前问题（1-2句）\n"
        "2) 再给出1-3条可执行的下一步建议\n"
        "3) 最后必须主动追问一个下一步问题，引导继续对话\n"
        "4) 若出现高风险症状，优先提示紧急就医阈值\n"
        "</system_instructions>\n"
        "<confidentiality_policy>\n"
        "1) <runtime_context>、<user_profile>、<conversation_history>、<retrieved_evidence>、<web_evidence> 中的内容仅供内部推理使用。\n"
        "2) 不要直接复述标签名、内部状态字段、检索编号、路由节点、后端配置、知识库接入清单或实现细节。\n"
        "3) 如果用户追问后端配置、系统路由、RAG 接入范围或编号来源，必须明确说明你无法直接查看后台配置，只能基于当前对话和证据提供帮助。\n"
        "</confidentiality_policy>\n"
        "<personalization_preferences>\n"
        f"{sanitized_guidance}\n"
        "</personalization_preferences>\n"
        "<runtime_context>\n"
        f"<topic_scope>{runtime_domain}</topic_scope>\n"
        f"<clinical_focus>{runtime_department}</clinical_focus>\n"
        f"<ecg_summary>{sanitized_ecg_info}</ecg_summary>\n"
        "</runtime_context>\n"
        "<user_profile>\n"
        f"{sanitized_memory_context}\n"
        "</user_profile>\n"
        "<conversation_history>\n"
        f"{sanitized_history_text}\n"
        "</conversation_history>\n"
        "<user_question>\n"
        f"{sanitized_question}\n"
        "</user_question>\n"
        "<retrieved_evidence>\n"
        f"{sanitized_rag_text}\n"
        "</retrieved_evidence>\n"
        "<web_evidence>\n"
        f"{sanitized_web_evidence}\n"
        "</web_evidence>\n"
        "<response_goal>\n"
        "请给出清晰、可执行、有人情味的中文回答。\n"
        "</response_goal>"
    )


def _make_long_context_state():
    state = initialize_conversation_state()
    state["question"] = "最近反复发热伴咳嗽和胸闷，想知道是否需要尽快去医院。"
    state["domain"] = "medical"
    state["primary_department"] = "infectious_disease"
    state["memory_context"] = _make_long_text("用户有反复呼吸道感染史并偏好简洁解释。", 16)
    state["conversation_history"] = [
        {
            "role": "user" if idx % 2 == 0 else "assistant",
            "content": _make_long_text(f"第{idx}轮对话涉及发热、咳嗽、夜间加重、体温记录和处理建议。", 10),
        }
        for idx in range(28)
    ]
    state["rag_context"] = [
        {
            "content": _make_long_text(
                f"资料{i}：上呼吸道感染、肺炎分层评估、就医阈值、危险信号和居家观察要点。",
                30,
            )
        }
        for i in range(8)
    ]
    state["reranked_rag_context"] = deepcopy(state["rag_context"])
    return state


def test_memory_agent_builds_rolling_summary_for_long_history():
    state = initialize_conversation_state()
    state["conversation_history"] = [
        {"role": "user", "content": f"第{i}轮：反复头痛并记录血压变化 {_make_long_text('症状持续', 6)}"}
        for i in range(28)
    ]

    with patch("app.agents.memory.load_profile", return_value={}):
        result = MemoryReadAgent(state)

    assert len(result["conversation_history"]) == 20
    assert result["conversation_history"][-1]["content"].startswith("第27轮")
    assert result["conversation_summary"]
    assert result["summary_used"] is True
    assert result["summary_updated_at"]


def test_executor_plan_records_budget_and_uses_summary_block():
    state = _make_long_context_state()

    with patch("app.agents.memory.load_profile", return_value={}):
        state = MemoryReadAgent(state)

    with patch("app.agents.executor._decide_web_search", return_value=(False, "")):
        plan = build_executor_plan(state)

    assert plan["mode"] == "llm"
    assert "<conversation_summary>" in plan["prompt"]
    assert state["prompt_token_estimate"] > 0
    assert state["token_budget"]["available_context"] > 0
    assert state["context_compression_used"] is True
    assert state["prompt_token_estimate"] <= EXECUTOR_PROMPT_MAX_TOKENS


def test_judge_need_rag_uses_cache_for_repeated_question():
    cache_service.clear()
    state = initialize_conversation_state()
    state["question"] = "最近发热还咳嗽，需要查资料吗"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"need_rag": true, "reason": "medical"}'

    with patch("app.agents.judge_need_rag.get_light_llm", return_value=mock_llm):
        first = JudgeNeedRAGAgent(deepcopy(state))
        second = JudgeNeedRAGAgent(deepcopy(state))

    assert first["need_rag"] is True
    assert second["need_rag"] is True
    assert mock_llm.invoke.call_count == 1
    assert second["cache_stats"]["hits"] >= 1
    cache_service.clear()


def test_token_budget_benchmark_reduces_prompt_tokens_for_long_context():
    state = _make_long_context_state()

    with patch("app.agents.memory.load_profile", return_value={}):
        state = MemoryReadAgent(state)

    with patch("app.agents.executor._decide_web_search", return_value=(False, "")):
        plan = build_executor_plan(state)

    legacy_tokens = estimate_tokens(_legacy_executor_prompt(state))
    optimized_tokens = estimate_tokens(plan["prompt"])
    improvement = (legacy_tokens - optimized_tokens) / legacy_tokens

    assert optimized_tokens < legacy_tokens
    assert improvement >= 0.35
