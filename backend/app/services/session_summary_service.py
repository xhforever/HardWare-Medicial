"""
MediGenius — services/session_summary_service.py
Rolling summary maintenance for long-running sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import threading
from typing import Any

from app.core.config import (
    HISTORY_HARD_LIMIT,
    SESSION_SUMMARY_ENABLED,
    SESSION_SUMMARY_USE_LLM,
    TOKEN_BUDGET_SUMMARY_TOKENS,
)
from app.core.logging_config import logger
from app.services.token_budget_service import truncate_text_to_budget
from app.tools.llm_client import coerce_response_text, get_light_llm, invoke_with_metrics

_summary_lock = threading.Lock()


def _normalize_line(content: str) -> str:
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    return truncate_text_to_budget(text, 48)


def _delta_lines(history: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in history[-8:]:
        content = _normalize_line(item.get("content", ""))
        if not content:
            continue
        prefix = "用户提到" if item.get("role") == "user" else "助手建议"
        lines.append(f"- {prefix}：{content}")
    return lines


def _summary_signature(existing_summary: str, delta_lines: list[str]) -> str:
    payload = {
        "existing_summary": existing_summary,
        "delta_lines": delta_lines,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _heuristic_summary(existing_summary: str, delta_lines: list[str]) -> str:
    parts = [existing_summary.strip()] if existing_summary.strip() else []
    if delta_lines:
        parts.append("\n".join(delta_lines))
    merged = "\n".join(part for part in parts if part)
    return truncate_text_to_budget(merged, TOKEN_BUDGET_SUMMARY_TOKENS)


def _llm_summary(
    state: dict[str, Any],
    existing_summary: str,
    delta_lines: list[str],
) -> tuple[str, str]:
    if not SESSION_SUMMARY_USE_LLM:
        return _heuristic_summary(existing_summary, delta_lines), "heuristic"

    llm = get_light_llm(
        tenant_id=state.get("tenant_id", "default"),
        user_id=state.get("user_id", "anonymous"),
    )
    if not llm:
        return _heuristic_summary(existing_summary, delta_lines), "heuristic"

    prompt = (
        "你负责维护医疗助手的滚动会话摘要。\n"
        "请把已有摘要与新增历史整理成 4 条以内的简体中文要点。\n"
        "只保留症状、时间线、已给建议、用户偏好和未解决问题。\n"
        "不要写内部标签，不要生成诊断结论。\n\n"
        f"已有摘要：\n{existing_summary or '暂无'}\n\n"
        f"新增历史：\n{chr(10).join(delta_lines) or '暂无'}\n"
    )
    try:
        raw = invoke_with_metrics(llm, prompt, node_name="session_summary", state=state)
        content = coerce_response_text(raw).strip()
        if content:
            return truncate_text_to_budget(content, TOKEN_BUDGET_SUMMARY_TOKENS), "light_llm"
    except Exception as exc:
        logger.warning("Session summary llm fallback used: %s", exc)
    return _heuristic_summary(existing_summary, delta_lines), "heuristic"


def refresh_conversation_summary(
    state: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    history = list(state.get("conversation_history") or [])
    existing_summary = str(state.get("conversation_summary") or "").strip()

    if not SESSION_SUMMARY_ENABLED:
        if len(history) > HISTORY_HARD_LIMIT:
            state["conversation_history"] = history[-HISTORY_HARD_LIMIT:]
        state["summary_used"] = False
        return state

    if len(history) <= HISTORY_HARD_LIMIT:
        state["conversation_history"] = history
        state["summary_used"] = bool(existing_summary)
        return state

    older_history = history[:-HISTORY_HARD_LIMIT]
    recent_history = history[-HISTORY_HARD_LIMIT:]
    delta_lines = _delta_lines(older_history)
    signature = _summary_signature(existing_summary, delta_lines)

    with _summary_lock:
        if not force and signature == state.get("summary_signature") and existing_summary:
            state["conversation_history"] = recent_history
            state["summary_used"] = True
            return state

    summary_text, summary_source = _llm_summary(state, existing_summary, delta_lines)
    if not summary_text:
        summary_text = _heuristic_summary(existing_summary, delta_lines)
        summary_source = "heuristic"

    with _summary_lock:
        state["conversation_summary"] = summary_text
        state["summary_source"] = summary_source
        state["summary_signature"] = signature
        state["summary_updated_at"] = datetime.now(timezone.utc).isoformat()
        state["conversation_history"] = recent_history
        state["summary_used"] = bool(summary_text)
    return state


def schedule_summary_refresh(state: dict[str, Any]) -> None:
    if not SESSION_SUMMARY_ENABLED:
        return
    if len(state.get("conversation_history") or []) <= HISTORY_HARD_LIMIT:
        return

    thread = threading.Thread(
        target=refresh_conversation_summary,
        args=(state,),
        kwargs={"force": True},
        daemon=True,
    )
    thread.start()
