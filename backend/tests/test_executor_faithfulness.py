"""Claim-level faithfulness baseline tests for executor answers."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.executor import ExecutorAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    AnswerEvidenceCoverageMetric,
    ChineseOutputMetric,
    ClaimFaithfulnessMetric,
    FollowUpQuestionMetric,
    NoInternalLeakageMetric,
)


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "eval_cases"
    / "executor_faithfulness_cases.json"
)
EXECUTOR_FAITHFULNESS_CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _build_executor_state(case_data: dict) -> dict:
    state = initialize_conversation_state()
    state["question"] = case_data["question"]
    state["domain"] = "medical"
    state["use_rag"] = True
    state["source"] = case_data["source"]
    state["primary_department"] = case_data.get("primary_department", "general_medical")
    state["department_candidates"] = [
        {"name": state["primary_department"], "score": 0.9}
    ]
    state["rag_context"] = list(case_data["rag_context"])
    return state


def _build_faithfulness_metrics(case_data: dict) -> list:
    return [
        ChineseOutputMetric(),
        FollowUpQuestionMetric(),
        NoInternalLeakageMetric(
            [
                "<system_instructions>",
                "<runtime_context>",
                "<conversation_history>",
                "<retrieved_evidence>",
                "<web_evidence>",
                "[RAG-",
                "[WEB-",
            ]
        ),
        ClaimFaithfulnessMetric(
            case_data["supported_claims"],
            forbidden_claims=case_data.get("forbidden_claims") or [],
            min_supported_claims=case_data.get("min_supported_claims"),
            min_coverage=case_data.get("min_coverage", 0.75),
            max_forbidden_claims=case_data.get("max_forbidden_claims", 0),
        ),
        AnswerEvidenceCoverageMetric(
            case_data["supported_claims"],
            threshold=case_data.get("min_coverage", 0.75),
        ),
    ]


def test_executor_faithfulness_gold_dataset_has_4_cases():
    assert len(EXECUTOR_FAITHFULNESS_CASES) == 4


@pytest.mark.parametrize(
    "case_data",
    EXECUTOR_FAITHFULNESS_CASES,
    ids=[case["case_id"] for case in EXECUTOR_FAITHFULNESS_CASES],
)
def test_executor_faithfulness_baseline(case_data):
    state = _build_executor_state(case_data)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = case_data["baseline_answer"]

    with patch("app.agents.executor.get_llm", return_value=mock_llm), patch(
        "app.agents.executor._decide_web_search", return_value=(False, "")
    ):
        result = ExecutorAgent(state)

    case = LLMTestCase(
        input=case_data["question"],
        actual_output=result["generation"],
    )
    assert_test(
        case,
        metrics=_build_faithfulness_metrics(case_data),
        run_async=False,
    )
    assert result["source"] == case_data["source"]
