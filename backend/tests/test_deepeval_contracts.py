"""Contract tests backed by pytest + deepeval.

These tests protect high-level user-facing response invariants so future
token-budget refactors do not silently regress core behavior.
"""

import os
import re
import sys
from unittest.mock import MagicMock, patch

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from app.agents.executor import ExecutorAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.services.greeting_service import GreetingService  # noqa: E402


class ChineseOutputMetric(BaseMetric):
    def __init__(self, min_chinese_chars: int = 12):
        self.min_chinese_chars = min_chinese_chars
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        self.score = 1.0 if chinese_count >= self.min_chinese_chars else 0.0
        self.reason = (
            f"detected {chinese_count} Chinese characters"
            if self.score
            else f"only detected {chinese_count} Chinese characters"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "ChineseOutputMetric"


class FollowUpQuestionMetric(BaseMetric):
    def __init__(self):
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True
        self._patterns = (
            "？",
            "?",
            "你希望我下一步",
            "你现在",
            "你最近",
            "是否",
            "有没有",
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        matched = [pattern for pattern in self._patterns if pattern in text]
        self.score = 1.0 if matched else 0.0
        self.reason = (
            f"matched follow-up markers: {matched}"
            if matched
            else "no follow-up question marker detected"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "FollowUpQuestionMetric"


class NoInternalLeakageMetric(BaseMetric):
    def __init__(self, forbidden_terms: list[str]):
        self.forbidden_terms = forbidden_terms
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        hits = [term for term in self.forbidden_terms if term in text]
        self.score = 1.0 if not hits else 0.0
        self.reason = (
            "no internal leakage markers found"
            if not hits
            else f"found: {hits}"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "NoInternalLeakageMetric"


def _assert_core_response_contract(question: str, answer: str) -> None:
    case = LLMTestCase(input=question, actual_output=answer)
    assert_test(
        case,
        metrics=[
            ChineseOutputMetric(),
            FollowUpQuestionMetric(),
            NoInternalLeakageMetric(
                forbidden_terms=[
                    "<system_instructions>",
                    "<runtime_context>",
                    "<conversation_history>",
                    "<retrieved_evidence>",
                    "<web_evidence>",
                    "<user_profile>",
                    "[RAG-",
                    "[WEB-",
                    "RAG-1",
                    "WEB-1",
                    "Patient:",
                    "Doctor:",
                    "No persistent memory context.",
                ]
            ),
        ],
        run_async=False,
    )


def test_executor_deepeval_contract_with_rag_context():
    state = initialize_conversation_state()
    state["question"] = "最近咳嗽、流鼻涕，应该怎么处理？"
    state["rag_context"] = [
        {
            "content": "普通上呼吸道感染多数可先补液休息，并观察体温与呼吸情况。",
        }
    ]
    state["source"] = "通用医疗知识库"

    with patch("app.agents.executor.get_llm") as mock_get_llm, patch(
        "app.agents.executor._decide_web_search", return_value=(False, "")
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "结合你目前描述，更像是上呼吸道感染或普通感冒。"
            "建议先补液休息、监测体温，如果高热持续或呼吸困难要及时就医。"
            "你现在有发热、咽痛或气喘吗？"
        )
        mock_get_llm.return_value = mock_llm

        result = ExecutorAgent(state)

    _assert_core_response_contract(state["question"], result["generation"])
    assert result["source"] == "通用医疗知识库"


def test_executor_deepeval_contract_without_llm():
    state = initialize_conversation_state()
    state["question"] = "这两天有点头晕，我该怎么办？"

    with patch("app.agents.executor.get_llm", return_value=None):
        result = ExecutorAgent(state)

    _assert_core_response_contract(state["question"], result["generation"])
    assert result["source"] == "System Message"


def test_greeting_deepeval_contract_for_empty_session():
    service = GreetingService()
    profile = {
        "preferences": {"language": "简体中文", "communication_style": "温和"},
        "current_context": {
            "last_ecg_diagnosis": "窦性心律",
            "last_ecg_risk_level": "low",
            "last_ecg_heart_rate": "72 bpm",
        },
    }

    with patch(
        "app.services.greeting_service.db_service.get_chat_history",
        return_value=[],
    ), patch(
        "app.services.greeting_service.db_service.save_message"
    ), patch(
        "app.services.greeting_service.load_profile",
        return_value=profile,
    ), patch(
        "app.services.greeting_service._fetch_weather",
        return_value="",
    ), patch(
        "app.services.greeting_service.get_light_llm",
        return_value=None,
    ):
        result = service.generate_greeting("sess-deepeval")

    _assert_core_response_contract("为空会话生成欢迎语", result["response"])
    assert result["source"] == "Welcome Concierge"
    assert result["success"] is True
