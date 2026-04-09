"""Tool usage capability tests using pytest + deepeval."""

import os
import sys
from unittest.mock import MagicMock, patch

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.executor import (  # noqa: E402
    ExecutorAgent,
    MAX_TOOL_CALLS,
    MAX_SAME_TOOL_REPEAT,
    _run_web_search,
)
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    JsonFieldsMatchMetric,
    ToolBudgetMetric,
    serialize_payload,
)


def test_executor_tool_usage_contract_for_temporal_question():
    state = initialize_conversation_state()
    state["question"] = "最新高血压指南有什么变化？"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "根据最新资料，重点仍然是长期控制血压。你现在血压大概多少？"

    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = [
        {"title": "高血压指南更新", "content": "最新指南强调长期血压管理与个体化治疗。"}
    ]

    with patch("app.agents.executor.get_llm", return_value=mock_llm), patch(
        "app.agents.executor.get_tavily_search", return_value=mock_tavily
    ), patch("app.agents.executor.WEB_SEARCH_ENABLED", True), patch(
        "app.agents.executor.WEB_SEARCH_USE_LLM_DECIDER", False
    ):
        result = ExecutorAgent(state)

    case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(
            {
                "tool_calls": result["tool_calls"],
                "tool_budget_used": result["tool_budget_used"],
                "source": result["source"],
            }
        ),
    )
    assert_test(
        case,
        metrics=[
            ToolBudgetMetric(MAX_TOOL_CALLS, MAX_SAME_TOOL_REPEAT),
            JsonFieldsMatchMetric({"source": "Current Medical Research & News"}),
        ],
        run_async=False,
    )


def test_executor_tool_usage_contract_skips_web_search_in_manual_scope():
    state = initialize_conversation_state()
    state["question"] = "最新偏头痛指南有什么变化？"
    state["selected_department"] = "neurology"
    state["selected_department_forced"] = True

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "在锁定专科模式下，我先基于本地资料回答。你最近头痛频率高吗？"

    with patch("app.agents.executor.get_llm", return_value=mock_llm), patch(
        "app.agents.executor.get_tavily_search"
    ) as mock_get_tavily, patch(
        "app.agents.executor.WEB_SEARCH_ENABLED", True
    ), patch(
        "app.agents.executor.WEB_SEARCH_USE_LLM_DECIDER", False
    ):
        result = ExecutorAgent(state)

    mock_get_tavily.assert_not_called()
    case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(
            {
                "tool_calls": result["tool_calls"],
                "tool_budget_used": result["tool_budget_used"],
            }
        ),
    )
    assert_test(
        case,
        metrics=[
            ToolBudgetMetric(MAX_TOOL_CALLS, MAX_SAME_TOOL_REPEAT),
            JsonFieldsMatchMetric(
                {
                    "tool_budget_used": 0,
                }
            ),
        ],
        run_async=False,
    )


def test_run_web_search_repeat_limit_contract():
    state = initialize_conversation_state()
    state["tool_calls"] = [{"tool": "web_search", "query": "最新指南"}]
    state["tool_budget_used"] = 1

    result = _run_web_search(state, "最新指南")
    assert result == ""
    assert state["tool_budget_used"] == 1
    assert len(state["tool_calls"]) == 1
