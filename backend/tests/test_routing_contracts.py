"""Routing capability tests using pytest + deepeval."""

import os
import sys
from unittest.mock import MagicMock, patch

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.judge_need_rag import JudgeNeedRAGAgent  # noqa: E402
from app.agents.medical_router import MedicalRouterAgent  # noqa: E402
from app.agents.planner import HealthConciergeAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import JsonFieldsMatchMetric, serialize_payload  # noqa: E402


def _assert_json_contract(question: str, payload: dict, expected_fields: dict) -> None:
    case = LLMTestCase(
        input=question,
        actual_output=serialize_payload(payload),
    )
    assert_test(case, [JsonFieldsMatchMetric(expected_fields)], run_async=False)


def test_health_concierge_routes_medical_question_to_medical_router():
    state = initialize_conversation_state()
    state["question"] = "我最近发烧 咳嗽，还想问退烧药怎么吃"

    with patch("app.agents.planner.get_light_llm", return_value=None):
        result = HealthConciergeAgent(state)

    _assert_json_contract(
        state["question"],
        {
            "domain": result["domain"],
            "current_tool": result["current_tool"],
            "safety_level": result["safety_level"],
        },
        {
            "domain": "medical",
            "current_tool": "medical_router",
            "safety_level": "SAFE",
        },
    )


def test_health_concierge_routes_greeting_to_general_judge():
    state = initialize_conversation_state()
    state["question"] = "你好"

    with patch("app.agents.planner.get_light_llm", return_value=None):
        result = HealthConciergeAgent(state)

    _assert_json_contract(
        state["question"],
        {
            "domain": result["domain"],
            "current_tool": result["current_tool"],
        },
        {
            "domain": "general",
            "current_tool": "judge_need_rag",
        },
    )


def test_health_concierge_manual_department_override_contract():
    state = initialize_conversation_state()
    state["question"] = "我想了解手麻平时如何护理"
    state["selected_department"] = "neurology"
    state["selected_department_forced"] = True

    with patch("app.agents.planner.get_light_llm", return_value=None):
        result = HealthConciergeAgent(state)

    _assert_json_contract(
        state["question"],
        {
            "domain": result["domain"],
            "current_tool": result["current_tool"],
            "primary_department": result["primary_department"],
        },
        {
            "domain": "medical",
            "current_tool": "query_rewriter",
            "primary_department": "neurology",
        },
    )


def test_health_concierge_llm_escalates_emergency_question():
    state = initialize_conversation_state()
    state["question"] = "我现在胸痛并且呼吸困难"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"safety_level":"EMERGENCY"}'

    with patch("app.agents.planner.get_light_llm", return_value=mock_llm):
        result = HealthConciergeAgent(state)

    _assert_json_contract(
        state["question"],
        {
            "safety_level": result["safety_level"],
            "current_tool": result["current_tool"],
        },
        {
            "safety_level": "EMERGENCY",
            "current_tool": "executor",
        },
    )


def test_medical_router_fallback_contract_for_hematology():
    state = initialize_conversation_state()
    state["question"] = "我血红蛋白低，经常头晕乏力"
    state["domain"] = "medical"
    state["use_rag"] = True

    with patch("app.agents.medical_router.get_light_llm", return_value=None):
        result = MedicalRouterAgent(state)

    _assert_json_contract(
        state["question"],
        {
            "primary_department": result["primary_department"],
            "current_tool": result["current_tool"],
        },
        {
            "primary_department": "hematology",
            "current_tool": "query_rewriter",
        },
    )
    assert result["department_candidates"]


def test_judge_need_rag_skips_retrieval_for_chitchat():
    state = initialize_conversation_state()
    state["question"] = "谢谢"

    result = JudgeNeedRAGAgent(state)
    _assert_json_contract(
        state["question"],
        {
            "need_rag": result["need_rag"],
            "search_query": result["search_query"],
        },
        {
            "need_rag": False,
            "search_query": None,
        },
    )
