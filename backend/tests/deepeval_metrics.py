"""Reusable local deepeval metrics for MediGenius contract tests."""

from __future__ import annotations
from collections import Counter
import json
import re
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from app.core.config import LLM_MODEL, OPENAI_WIRE_API
from app.core.logging_config import logger
from app.tools.llm_client import _resolve_llm_config, coerce_response_text


def serialize_payload(payload: Any) -> str:
    """Serialize a payload to compact JSON for deepeval test cases."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text or "")
    except Exception:
        return None


def _lookup_path(payload: Any, path: str) -> Any:
    current = payload
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def _extract_eval_terms(text: str) -> list[str]:
    if not text:
        return []

    terms = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower()):
        cleaned = token.strip()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        terms.append(cleaned)
    return terms


def build_project_chat_llm(
    *,
    tenant_id: str = "default",
    user_id: str = "anonymous",
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """Build a LangChain chat model from the project's runtime config."""
    cfg = _resolve_llm_config(tenant_id, user_id)
    api_key = cfg.get("api_key")
    model = cfg.get("model") or LLM_MODEL
    base_url = cfg.get("base_url")

    if not api_key:
        logger.warning(
            "Project eval LLM unavailable because OPENAI_API_KEY is not configured"
        )
        return None

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        logger.warning("langchain_openai unavailable for project eval LLM: %s", exc)
        return None

    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }
    if OPENAI_WIRE_API == "responses":
        kwargs["use_responses_api"] = True
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def _build_schema_fallback_prompt(prompt: Any, schema: type[BaseModel]) -> str:
    if isinstance(prompt, str):
        prompt_text = prompt
    else:
        prompt_text = json.dumps(prompt, ensure_ascii=False, default=str)

    schema_text = json.dumps(
        schema.model_json_schema(), ensure_ascii=False, sort_keys=True
    )
    return (
        f"{prompt_text}\n\n"
        "请严格输出一个 JSON 对象，不要输出额外解释、Markdown 或代码块。\n"
        f"JSON Schema:\n{schema_text}"
    )


class ProjectDeepEvalLLM(DeepEvalBaseLLM):
    """deepeval adapter that reuses MediGenius runtime LLM configuration."""

    def __init__(
        self,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        cfg = _resolve_llm_config(tenant_id, user_id)
        self._configured_model_name = str(cfg.get("model") or LLM_MODEL)
        super().__init__(model=self._configured_model_name)

    def load_model(self):
        llm = build_project_chat_llm(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if llm is None:
            raise RuntimeError("Project eval LLM is unavailable")
        return llm

    def generate(self, prompt: Any, schema: type[BaseModel] | None = None):
        if schema is None:
            return coerce_response_text(self.model.invoke(prompt))

        try:
            structured_llm = self.model.with_structured_output(schema)
            return structured_llm.invoke(prompt)
        except Exception as exc:
            logger.warning(
                "ProjectDeepEvalLLM structured output failed for %s: %s",
                self.get_model_name(),
                exc,
            )
            fallback_prompt = _build_schema_fallback_prompt(prompt, schema)
            return coerce_response_text(self.model.invoke(fallback_prompt))

    async def a_generate(self, prompt: Any, schema: type[BaseModel] | None = None):
        if schema is None:
            return coerce_response_text(await self.model.ainvoke(prompt))

        try:
            structured_llm = self.model.with_structured_output(schema)
            return await structured_llm.ainvoke(prompt)
        except Exception as exc:
            logger.warning(
                "ProjectDeepEvalLLM async structured output failed for %s: %s",
                self.get_model_name(),
                exc,
            )
            fallback_prompt = _build_schema_fallback_prompt(prompt, schema)
            return coerce_response_text(await self.model.ainvoke(fallback_prompt))

    def get_model_name(self, *args, **kwargs) -> str:
        return self._configured_model_name

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True


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
            else f"found leakage markers: {hits}"
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


class JsonFieldsMatchMetric(BaseMetric):
    """Check that JSON payload fields exactly match expected values."""

    def __init__(self, expected_fields: dict[str, Any]):
        self.expected_fields = expected_fields
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        if not isinstance(payload, dict):
            self.score = 0.0
            self.reason = "actual_output is not a JSON object"
            self.success = False
            return self.score

        mismatches = []
        for path, expected_value in self.expected_fields.items():
            actual_value = _lookup_path(payload, path)
            if actual_value != expected_value:
                mismatches.append(
                    f"{path}: expected={expected_value!r}, actual={actual_value!r}"
                )

        self.score = 1.0 if not mismatches else 0.0
        self.reason = "all expected fields matched" if not mismatches else "; ".join(mismatches)
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
        return "JsonFieldsMatchMetric"


class ContainsTermsMetric(BaseMetric):
    def __init__(self, required_terms: list[str]):
        self.required_terms = required_terms
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = (test_case.actual_output or "").lower()
        missing = [term for term in self.required_terms if term.lower() not in text]
        self.score = 1.0 if not missing else 0.0
        self.reason = (
            "all required terms found"
            if not missing
            else f"missing required terms: {missing}"
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
        return "ContainsTermsMetric"


class GoldQueryAlignmentMetric(BaseMetric):
    """Check whether a retrieval query preserves the core terms of a gold query."""

    def __init__(self, gold_queries: list[str], threshold: float = 0.75):
        self.gold_queries = gold_queries
        self.threshold = threshold
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_text = (test_case.actual_output or "").lower()
        if not actual_text.strip():
            self.score = 0.0
            self.reason = "actual_output is empty"
            self.success = False
            return self.score

        best_query = ""
        best_terms: list[str] = []
        best_matched: list[str] = []
        best_score = 0.0

        for gold_query in self.gold_queries:
            gold_terms = _extract_eval_terms(gold_query)
            if not gold_terms:
                continue

            matched_terms = [term for term in gold_terms if term in actual_text]
            coverage = len(matched_terms) / len(gold_terms)
            if coverage > best_score:
                best_query = gold_query
                best_terms = gold_terms
                best_matched = matched_terms
                best_score = coverage

        missing_terms = [term for term in best_terms if term not in best_matched]
        self.score = best_score
        self.reason = (
            f"best gold coverage={best_score:.2f} for query={best_query!r}; "
            f"matched={best_matched}; missing={missing_terms}"
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
        return "GoldQueryAlignmentMetric"


class NoAnswerStyleMetric(BaseMetric):
    def __init__(self):
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True
        self._answer_markers = (
            "建议",
            "应该",
            "可以先",
            "请先",
            "及时就医",
            "最好",
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        hits = [marker for marker in self._answer_markers if marker in text]
        self.score = 1.0 if not hits else 0.0
        self.reason = (
            "rewrite stays retrieval-oriented"
            if not hits
            else f"detected answer-style markers: {hits}"
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
        return "NoAnswerStyleMetric"


class JsonFieldContainsTermsMetric(BaseMetric):
    def __init__(self, field_path: str, required_terms: list[str]):
        self.field_path = field_path
        self.required_terms = required_terms
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        value = _lookup_path(payload, self.field_path) if payload is not None else None
        text = str(value or "").lower()
        missing = [term for term in self.required_terms if term.lower() not in text]
        self.score = 1.0 if not missing else 0.0
        self.reason = (
            "all required terms present in field"
            if not missing
            else f"missing terms in {self.field_path}: {missing}"
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
        return "JsonFieldContainsTermsMetric"


class JsonListTopFieldMetric(BaseMetric):
    def __init__(
        self,
        field_path: str,
        expected_value: Any,
        *,
        list_path: str | None = None,
        top_index: int = 0,
    ):
        self.field_path = field_path
        self.expected_value = expected_value
        self.list_path = list_path
        self.top_index = top_index
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        items = payload if self.list_path is None else _lookup_path(payload, self.list_path)
        if not isinstance(items, list) or len(items) <= self.top_index:
            self.score = 0.0
            self.reason = "target list missing or shorter than expected"
            self.success = False
            return self.score

        actual_value = _lookup_path(items[self.top_index], self.field_path)
        self.score = 1.0 if actual_value == self.expected_value else 0.0
        self.reason = (
            "top field matched expected value"
            if self.score
            else (
                f"expected top {self.field_path}={self.expected_value!r}, "
                f"got {actual_value!r}"
            )
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
        return "JsonListTopFieldMetric"


class JsonListContainsTermsMetric(BaseMetric):
    def __init__(
        self,
        field_path: str,
        required_terms: list[str],
        *,
        list_path: str | None = None,
        min_matches: int = 1,
    ):
        self.field_path = field_path
        self.required_terms = required_terms
        self.list_path = list_path
        self.min_matches = min_matches
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        items = payload if self.list_path is None else _lookup_path(payload, self.list_path)
        if not isinstance(items, list):
            self.score = 0.0
            self.reason = "target list missing"
            self.success = False
            return self.score

        matches = 0
        for item in items:
            text = str(_lookup_path(item, self.field_path) or "")
            if all(term in text for term in self.required_terms):
                matches += 1

        self.score = 1.0 if matches >= self.min_matches else 0.0
        self.reason = (
            f"found {matches} matching list entries"
            if self.score
            else (
                f"only found {matches} matching entries, "
                f"need at least {self.min_matches}"
            )
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
        return "JsonListContainsTermsMetric"


class ToolBudgetMetric(BaseMetric):
    def __init__(self, max_tool_calls: int, max_same_tool_repeat: int):
        self.max_tool_calls = max_tool_calls
        self.max_same_tool_repeat = max_same_tool_repeat
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        if not isinstance(payload, dict):
            self.score = 0.0
            self.reason = "actual_output is not a JSON object"
            self.success = False
            return self.score

        tool_calls = payload.get("tool_calls") or []
        tool_budget_used = int(payload.get("tool_budget_used") or 0)
        tool_counts = Counter(
            call.get("tool")
            for call in tool_calls
            if isinstance(call, dict) and call.get("tool")
        )

        violations = []
        if len(tool_calls) > self.max_tool_calls:
            violations.append(
                f"tool_calls length {len(tool_calls)} > {self.max_tool_calls}"
            )
        if tool_budget_used > self.max_tool_calls:
            violations.append(
                f"tool_budget_used {tool_budget_used} > {self.max_tool_calls}"
            )
        for tool_name, count in tool_counts.items():
            if count > self.max_same_tool_repeat:
                violations.append(
                    f"tool {tool_name} repeated {count} times > {self.max_same_tool_repeat}"
                )

        self.score = 1.0 if not violations else 0.0
        self.reason = "tool budget respected" if not violations else "; ".join(violations)
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
        return "ToolBudgetMetric"


class HistoryContainsMetric(BaseMetric):
    def __init__(
        self,
        required_terms: list[str],
        *,
        list_path: str = "conversation_history",
    ):
        self.required_terms = required_terms
        self.list_path = list_path
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        payload = _parse_json_payload(test_case.actual_output or "")
        history = _lookup_path(payload, self.list_path) if payload is not None else None
        if not isinstance(history, list):
            self.score = 0.0
            self.reason = "conversation history missing"
            self.success = False
            return self.score

        combined = "\n".join(str(item.get("content") or "") for item in history if isinstance(item, dict))
        missing = [term for term in self.required_terms if term not in combined]
        self.score = 1.0 if not missing else 0.0
        self.reason = (
            "all required history terms retained"
            if not missing
            else f"missing history terms: {missing}"
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
        return "HistoryContainsMetric"


class HighRiskEscalationMetric(BaseMetric):
    def __init__(self):
        self.threshold = 1.0
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True
        self.required_terms = ("急诊", "急救", "立即")

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        hits = [term for term in self.required_terms if term in text]
        self.score = 1.0 if hits else 0.0
        self.reason = (
            f"found escalation terms: {hits}"
            if hits
            else "missing emergency escalation language"
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
        return "HighRiskEscalationMetric"
