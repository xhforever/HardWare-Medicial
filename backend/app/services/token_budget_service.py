"""
MediGenius — services/token_budget_service.py
Token estimation and budget-driven context compression helpers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import (
    EXECUTOR_COMPLETION_RESERVE_TOKENS,
    EXECUTOR_PROMPT_MAX_TOKENS,
    TOKEN_BUDGET_ENABLED,
    TOKEN_BUDGET_MIN_SECTION_TOKENS,
    TOKEN_BUDGET_PROFILE_TOKENS,
    TOKEN_BUDGET_RAG_TOKENS,
    TOKEN_BUDGET_RECENT_HISTORY_TOKENS,
    TOKEN_BUDGET_SUMMARY_TOKENS,
    TOKEN_BUDGET_WEB_TOKENS,
)


@lru_cache(maxsize=8)
def _get_encoder(model: str = ""):
    try:
        import tiktoken
    except Exception:
        return None

    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model: str = "") -> int:
    payload = str(text or "")
    if not payload:
        return 0

    encoder = _get_encoder(model)
    if encoder is None:
        # Conservative fallback for mixed Chinese/English medical text.
        return max(1, len(payload) // 3)
    return len(encoder.encode(payload))


def truncate_text_to_budget(
    text: str,
    max_tokens: int,
    *,
    model: str = "",
    keep_tail: bool = False,
    suffix: str = "...",
) -> str:
    payload = str(text or "").strip()
    if not payload or max_tokens <= 0:
        return ""

    if estimate_tokens(payload, model=model) <= max_tokens:
        return payload

    encoder = _get_encoder(model)
    if encoder is None:
        approx_chars = max(1, max_tokens * 3)
        if keep_tail:
            return f"{suffix}{payload[-approx_chars:]}"
        return f"{payload[:approx_chars]}{suffix}"

    suffix_tokens = encoder.encode(suffix) if suffix else []
    allowed = max(1, max_tokens - len(suffix_tokens))
    raw_tokens = encoder.encode(payload)
    if keep_tail:
        clipped = encoder.decode(raw_tokens[-allowed:])
        return f"{suffix}{clipped}"
    clipped = encoder.decode(raw_tokens[:allowed])
    return f"{clipped}{suffix}"


def _format_history_line(item: dict[str, Any], *, model: str, per_message_tokens: int) -> str:
    role = "用户" if item.get("role") == "user" else "助手"
    content = truncate_text_to_budget(
        str(item.get("content", "")),
        per_message_tokens,
        model=model,
    )
    return f"{role}: {content}".strip()


def render_recent_history_text(
    history: list[dict[str, Any]],
    max_tokens: int,
    *,
    model: str = "",
    per_message_tokens: int = 96,
) -> str:
    if not history or max_tokens <= 0:
        return ""

    selected: list[str] = []
    used_tokens = 0
    for item in reversed(history):
        line = _format_history_line(item, model=model, per_message_tokens=per_message_tokens)
        if not line:
            continue
        line_tokens = estimate_tokens(line, model=model)
        if selected and used_tokens + line_tokens > max_tokens:
            break
        if not selected and line_tokens > max_tokens:
            line = truncate_text_to_budget(line, max_tokens, model=model, keep_tail=True)
            line_tokens = estimate_tokens(line, model=model)
        if line_tokens > max_tokens:
            continue
        selected.append(line)
        used_tokens += line_tokens

    selected.reverse()
    return "\n".join(selected)


def render_summary_text(summary: str, max_tokens: int, *, model: str = "") -> str:
    return truncate_text_to_budget(summary, max_tokens, model=model)


def render_evidence_text(
    chunks: list[dict[str, Any]],
    max_tokens: int,
    *,
    model: str = "",
    per_chunk_tokens: int = 180,
    max_chunks: int = 6,
) -> str:
    if not chunks or max_tokens <= 0:
        return ""

    selected: list[str] = []
    used_tokens = 0
    for chunk in chunks[:max_chunks]:
        content = truncate_text_to_budget(
            str(chunk.get("content", "")),
            per_chunk_tokens,
            model=model,
        )
        if not content:
            continue
        chunk_tokens = estimate_tokens(content, model=model)
        if selected and used_tokens + chunk_tokens > max_tokens:
            break
        if not selected and chunk_tokens > max_tokens:
            content = truncate_text_to_budget(content, max_tokens, model=model)
            chunk_tokens = estimate_tokens(content, model=model)
        if chunk_tokens > max_tokens:
            continue
        selected.append(content)
        used_tokens += chunk_tokens
    return "\n\n".join(selected)


def allocate_context_budgets(
    section_texts: dict[str, str],
    *,
    fixed_tokens: int,
    prompt_max_tokens: int = EXECUTOR_PROMPT_MAX_TOKENS,
    completion_reserve_tokens: int = EXECUTOR_COMPLETION_RESERVE_TOKENS,
    min_section_tokens: int = TOKEN_BUDGET_MIN_SECTION_TOKENS,
) -> dict[str, int]:
    desired_caps = {
        "user_profile": TOKEN_BUDGET_PROFILE_TOKENS,
        "conversation_summary": TOKEN_BUDGET_SUMMARY_TOKENS,
        "conversation_history": TOKEN_BUDGET_RECENT_HISTORY_TOKENS,
        "retrieved_evidence": TOKEN_BUDGET_RAG_TOKENS,
        "web_evidence": TOKEN_BUDGET_WEB_TOKENS,
    }
    active_sections = {
        name: text
        for name, text in section_texts.items()
        if str(text or "").strip()
    }
    available = max(0, prompt_max_tokens - completion_reserve_tokens - fixed_tokens)
    budgets = {
        "available_context": available,
        "completion_reserve": completion_reserve_tokens,
        "fixed_prompt": fixed_tokens,
        "prompt_max": prompt_max_tokens,
    }

    if not TOKEN_BUDGET_ENABLED or not active_sections:
        for name, text in section_texts.items():
            budgets[name] = estimate_tokens(text)
        return budgets

    desired = {
        name: min(estimate_tokens(text), desired_caps.get(name, estimate_tokens(text)))
        for name, text in active_sections.items()
    }
    if not desired:
        return budgets

    names = list(desired.keys())
    if available <= 0:
        for name in names:
            budgets[name] = 0
        return budgets

    if len(names) * min_section_tokens >= available:
        base = max(1, available // len(names))
        for name in names:
            budgets[name] = min(desired[name], base)
        leftover = max(0, available - sum(budgets[name] for name in names))
    else:
        for name in names:
            budgets[name] = min(desired[name], min_section_tokens)
        leftover = max(0, available - sum(budgets[name] for name in names))

    while leftover > 0:
        candidates = [name for name in names if budgets[name] < desired[name]]
        if not candidates:
            break
        total_weight = sum(desired_caps.get(name, desired[name]) for name in candidates)
        progressed = False
        for idx, name in enumerate(candidates):
            remaining_need = desired[name] - budgets[name]
            if remaining_need <= 0:
                continue
            if idx == len(candidates) - 1:
                delta = min(remaining_need, leftover)
            else:
                weight = desired_caps.get(name, desired[name])
                delta = min(
                    remaining_need,
                    max(1, int(leftover * weight / max(1, total_weight))),
                )
            if delta <= 0:
                continue
            budgets[name] += delta
            leftover -= delta
            progressed = True
            if leftover <= 0:
                break
        if not progressed:
            break

    for name in section_texts:
        budgets.setdefault(name, 0)
    return budgets


def compress_context_sections(
    *,
    history: list[dict[str, Any]],
    summary: str,
    memory_context: str,
    rag_context: list[dict[str, Any]],
    web_evidence: str,
    fixed_tokens: int,
    model: str = "",
) -> tuple[dict[str, str], dict[str, int], bool]:
    raw_sections = {
        "user_profile": memory_context,
        "conversation_summary": summary,
        "conversation_history": "\n".join(
            _format_history_line(item, model=model, per_message_tokens=96)
            for item in history
        ).strip(),
        "retrieved_evidence": "\n\n".join(
            str(item.get("content", "")).strip() for item in rag_context if item.get("content")
        ).strip(),
        "web_evidence": web_evidence,
    }
    budgets = allocate_context_budgets(raw_sections, fixed_tokens=fixed_tokens)
    compressed = {
        "user_profile": truncate_text_to_budget(
            memory_context,
            budgets.get("user_profile", 0),
            model=model,
        ),
        "conversation_summary": render_summary_text(
            summary,
            budgets.get("conversation_summary", 0),
            model=model,
        ),
        "conversation_history": render_recent_history_text(
            history,
            budgets.get("conversation_history", 0),
            model=model,
        ),
        "retrieved_evidence": render_evidence_text(
            rag_context,
            budgets.get("retrieved_evidence", 0),
            model=model,
        ),
        "web_evidence": truncate_text_to_budget(
            web_evidence,
            budgets.get("web_evidence", 0),
            model=model,
        ),
    }
    compression_used = any(
        estimate_tokens(raw_sections.get(name, ""), model=model)
        > estimate_tokens(compressed.get(name, ""), model=model)
        for name in raw_sections
        if raw_sections.get(name)
    )
    return compressed, budgets, compression_used
