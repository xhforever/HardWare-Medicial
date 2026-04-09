"""Memory and multi-turn capability tests using pytest + deepeval."""

import os
import sys
from unittest.mock import patch

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.memory import MemoryReadAgent, MemoryWriteAsyncAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    HistoryContainsMetric,
    JsonFieldContainsTermsMetric,
    JsonFieldsMatchMetric,
    serialize_payload,
)


def test_memory_read_contract_trims_history_and_loads_profile():
    state = initialize_conversation_state()
    state["session_id"] = "sess-memory-1"
    state["conversation_history"] = [
        {"role": "user", "content": f"消息{i}"}
        for i in range(25)
    ]

    profile = {
        "preferences": {
            "preferred_name": "王女士",
            "communication_style": "concise",
        },
        "current_context": {"symptom": "咳嗽"},
    }

    with patch("app.agents.memory.load_profile", return_value=profile):
        result = MemoryReadAgent(state)

    payload = {
        "history_len": len(result["conversation_history"]),
        "preferred_name": result["user_preferences"].get("preferred_name"),
        "memory_context": result["memory_context"],
    }
    case = LLMTestCase(
        input="读取长期记忆",
        actual_output=serialize_payload(payload),
    )
    assert_test(
        case,
        metrics=[
            JsonFieldsMatchMetric(
                {
                    "history_len": 20,
                    "preferred_name": "王女士",
                }
            ),
            JsonFieldContainsTermsMetric("memory_context", ["王女士", "咳嗽"]),
        ],
        run_async=False,
    )


def test_memory_write_async_contract_schedules_profile_update():
    state = initialize_conversation_state()
    state["session_id"] = "sess-memory-2"
    state["question"] = "我叫李先生，希望回答简洁一点"
    state["generation"] = "李先生，我会尽量回答得更简洁。你最近还有哪里不舒服？"

    with patch("app.agents.memory.schedule_profile_update") as mock_schedule:
        MemoryWriteAsyncAgent(state)

    mock_schedule.assert_called_once_with(
        "sess-memory-2",
        state["question"],
        state["generation"],
        tenant_id="default",
        user_id="anonymous",
    )


def test_chat_service_bootstrap_contract_retains_previous_turns():
    service = ChatService()
    history = [
        {"role": "assistant", "content": "欢迎回来", "source": "Welcome Concierge"},
        {"role": "user", "content": "昨天开始发热 38.5 度"},
        {"role": "assistant", "content": "建议继续监测体温"},
    ]

    restored = service._load_persisted_history(
        "sess-memory-3",
        tenant_id="default",
        user_id="anonymous",
    ) if False else None
    assert restored is None  # keep linters quiet; real assertion uses patch below

    with patch(
        "app.services.chat_service.db_service.get_chat_history",
        return_value=history,
    ):
        restored = service._load_persisted_history(
            "sess-memory-3",
            tenant_id="default",
            user_id="anonymous",
        )

    case = LLMTestCase(
        input="恢复多轮历史",
        actual_output=serialize_payload({"conversation_history": restored}),
    )
    assert_test(
        case,
        metrics=[
            HistoryContainsMetric(["欢迎回来", "昨天开始发热 38.5 度"]),
        ],
        run_async=False,
    )
    assert restored[0]["source"] == "Welcome Concierge"
