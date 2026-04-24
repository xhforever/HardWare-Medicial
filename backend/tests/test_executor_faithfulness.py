"""LLM-judge faithfulness baseline tests for executor answers."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.executor import ExecutorAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    ChineseOutputMetric,
    FollowUpQuestionMetric,
    NoInternalLeakageMetric,
    ProjectDeepEvalLLM,
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


def _build_context_texts(case_data: dict) -> list[str]:
    return [
        item["content"] for item in case_data["rag_context"] if item.get("content")
    ]


def _build_faithfulness_metrics(case_data: dict, judge_llm: ProjectDeepEvalLLM) -> list:
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
        FaithfulnessMetric(
            threshold=case_data.get("faithfulness_threshold", 0.65),
            model=judge_llm,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
        HallucinationMetric(
            threshold=case_data.get("hallucination_threshold", 0.33),
            model=judge_llm,
            include_reason=True,
            async_mode=False,
            verbose_mode=False,
        ),
    ]


def test_executor_faithfulness_gold_dataset_has_4_cases():
    assert len(EXECUTOR_FAITHFULNESS_CASES) == 4


@pytest.fixture(scope="module")
def judge_llm():
    try:
        llm = ProjectDeepEvalLLM(temperature=0.0, max_tokens=1024)
    except RuntimeError:
        pytest.skip("Project judge LLM is unavailable for faithfulness baseline")

    class _ProbeSchema(BaseModel):
        status: str

    try:
        llm.generate("请输出一个 status 字段，值为 ok。", schema=_ProbeSchema)
    except Exception as exc:
        pytest.skip(
            f"Project judge LLM is unavailable for faithfulness baseline: {exc}"
        )

    return llm


@pytest.mark.parametrize(
    "case_data",
    EXECUTOR_FAITHFULNESS_CASES,
    ids=[case["case_id"] for case in EXECUTOR_FAITHFULNESS_CASES],
)
def test_executor_faithfulness_baseline(case_data, judge_llm):
    state = _build_executor_state(case_data)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = case_data["baseline_answer"]

    with patch("app.agents.executor.get_llm", return_value=mock_llm), patch(
        "app.agents.executor._decide_web_search", return_value=(False, "")
    ):
        result = ExecutorAgent(state)

    context_texts = _build_context_texts(case_data)
    case = LLMTestCase(
        input=case_data["question"],
        actual_output=result["generation"],
        context=context_texts,
        retrieval_context=context_texts,
        name=case_data["case_id"],
        comments=case_data.get("judge_focus"),
    )
    assert_test(
        case,
        metrics=_build_faithfulness_metrics(case_data, judge_llm),
        run_async=False,
    )
    assert result["source"] == case_data["source"]
