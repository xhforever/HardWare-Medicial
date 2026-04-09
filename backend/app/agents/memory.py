"""
MediGenius — agents/memory.py
Memory agents:
  - MemoryReadAgent: trim recent history + load profile context
  - MemoryWriteAsyncAgent: async profile update after final answer
"""

from app.core.config import HISTORY_HARD_LIMIT
from app.core.state import AgentState, append_flow_trace
from app.services.profile_service import (
    load_profile,
    render_profile_as_text,
    schedule_profile_update,
)
from app.services.session_summary_service import (
    refresh_conversation_summary,
    schedule_summary_refresh,
)


def MemoryReadAgent(state: AgentState) -> AgentState:
    """Trim history and load persistent profile context into state."""
    append_flow_trace(state, "memory_read")
    history = state.get("conversation_history", [])
    if len(history) > HISTORY_HARD_LIMIT:
        state["conversation_history"] = history
        refresh_conversation_summary(state)
    else:
        state["conversation_history"] = history[-HISTORY_HARD_LIMIT:]
        state["summary_used"] = bool(state.get("conversation_summary"))

    session_id = state.get("session_id", "")
    tenant_id = state.get("tenant_id", "default")
    user_id = state.get("user_id", "anonymous")
    profile = load_profile(
        session_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    state["memory_context"] = render_profile_as_text(profile)
    state["user_preferences"] = profile.get("preferences") or {}

    return state


def MemoryWriteAsyncAgent(state: AgentState) -> AgentState:
    """Schedule asynchronous profile updates without blocking main response path."""
    append_flow_trace(state, "memory_write_async")
    session_id = state.get("session_id", "")
    tenant_id = state.get("tenant_id", "default")
    user_id = state.get("user_id", "anonymous")
    question = state.get("question", "")
    answer = state.get("generation", "")

    if session_id and question and answer:
        schedule_profile_update(
            session_id,
            question,
            answer,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    schedule_summary_refresh(state)

    return state


# Backward-compatible alias
MemoryAgent = MemoryReadAgent
