"""Query rewriting capability tests using pytest + deepeval."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.query_rewriter import QueryRewriterAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    ContainsTermsMetric,
    GoldQueryAlignmentMetric,
    JsonFieldContainsTermsMetric,
    JsonFieldsMatchMetric,
    NoAnswerStyleMetric,
    serialize_payload,
)


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "eval_cases"
    / "query_rewriter_gold_cases.json"
)
QUERY_REWRITER_GOLD_CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _build_gold_case_state(case_data: dict) -> dict:
    state = initialize_conversation_state()
    state["question"] = case_data["user_query"]
    state["domain"] = "medical"
    state["use_rag"] = True

    expected_scope = case_data["expected_scope"]
    if case_data["rewrite_mode"] == "manual_fast_path":
        state["selected_department"] = expected_scope
        state["selected_department_forced"] = True
    else:
        state["primary_department"] = expected_scope
        state["department_candidates"] = [{"name": expected_scope, "score": 0.9}]

    return state


def test_query_rewriter_llm_contract_keeps_medical_terms():
    state = initialize_conversation_state()
    state["question"] = "血红蛋白低 头晕 乏力"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "hematology"
    state["department_candidates"] = [{"name": "hematology", "score": 0.9}]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"retrieval_query":"贫血 血红蛋白低 头晕 乏力",'
        '"department_queries":{"hematology":"血液科 贫血 头晕 乏力"},'
        '"rewrite_reason":"llm rewrite"}'
    )

    with patch("app.agents.query_rewriter.get_light_llm", return_value=mock_llm):
        result = QueryRewriterAgent(state)

    case = LLMTestCase(
        input=state["question"],
        actual_output=result["retrieval_query"],
    )
    assert_test(
        case,
        metrics=[
            ContainsTermsMetric(["贫血", "头晕", "乏力"]),
            NoAnswerStyleMetric(),
        ],
        run_async=False,
    )

    detail_case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(result),
    )
    assert_test(
        detail_case,
        metrics=[
            JsonFieldContainsTermsMetric(
                "department_queries.hematology",
                ["血液科", "贫血"],
            ),
        ],
        run_async=False,
    )


def test_query_rewriter_manual_department_fast_path_contract():
    state = initialize_conversation_state()
    state["question"] = "头晕 头痛 视物旋转"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["selected_department"] = "neurology"
    state["selected_department_forced"] = True

    with patch("app.agents.query_rewriter.get_light_llm") as mock_get_light_llm:
        result = QueryRewriterAgent(state)

    mock_get_light_llm.assert_not_called()

    case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(result),
    )
    assert_test(
        case,
        metrics=[
            JsonFieldsMatchMetric(
                {"rewrite_reason": "manual department fast-path"}
            ),
            JsonFieldContainsTermsMetric(
                "department_queries.neurology",
                ["神经内科"],
            ),
        ],
        run_async=False,
    )


def test_query_rewriter_disabled_returns_original_question():
    state = initialize_conversation_state()
    state["question"] = "最近头痛怎么办"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "neurology"
    state["department_candidates"] = [{"name": "neurology", "score": 0.9}]

    with patch("app.agents.query_rewriter.QUERY_REWRITER_ENABLED", False):
        result = QueryRewriterAgent(state)

    assert result["retrieval_query"] == state["question"]
    assert result["rewrite_reason"] == "query rewriter disabled by config"


def test_query_rewriter_heuristic_mode_stays_retrieval_oriented():
    state = initialize_conversation_state()
    state["question"] = "血糖高 多饮 多尿"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.8}]

    with patch("app.agents.query_rewriter.QUERY_REWRITER_USE_LLM", False):
        result = QueryRewriterAgent(state)

    case = LLMTestCase(
        input=state["question"],
        actual_output=result["retrieval_query"],
    )
    assert_test(
        case,
        metrics=[
            ContainsTermsMetric(["血糖高", "多饮", "多尿"]),
            NoAnswerStyleMetric(),
        ],
        run_async=False,
    )


def test_query_rewriter_gold_dataset_has_30_cases():
    assert len(QUERY_REWRITER_GOLD_CASES) == 30


@pytest.mark.parametrize(
    "case_data",
    QUERY_REWRITER_GOLD_CASES,
    ids=[case["case_id"] for case in QUERY_REWRITER_GOLD_CASES],
)
def test_query_rewriter_gold_dataset_contract(case_data):
    state = _build_gold_case_state(case_data)

    with patch("app.agents.query_rewriter.QUERY_REWRITER_USE_LLM", False), patch(
        "app.agents.query_rewriter.get_light_llm"
    ) as mock_get_light_llm:
        result = QueryRewriterAgent(state)

    mock_get_light_llm.assert_not_called()

    retrieval_case = LLMTestCase(
        input=case_data["user_query"],
        actual_output=result["retrieval_query"],
    )
    assert_test(
        retrieval_case,
        metrics=[
            ContainsTermsMetric(case_data["must_keep_terms"]),
            GoldQueryAlignmentMetric(case_data["gold_retrieval_queries"]),
            NoAnswerStyleMetric(),
        ],
        run_async=False,
    )

    detail_case = LLMTestCase(
        input=case_data["user_query"],
        actual_output=serialize_payload(result),
    )
    assert_test(
        detail_case,
        metrics=[
            JsonFieldContainsTermsMetric(
                f"department_queries.{case_data['expected_scope']}",
                case_data["required_department_query_terms"],
            ),
            JsonFieldsMatchMetric(
                {"rewrite_reason": case_data["expected_rewrite_reason"]}
            ),
        ],
        run_async=False,
    )
