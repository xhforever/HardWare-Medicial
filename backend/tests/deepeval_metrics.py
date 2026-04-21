"""Reusable local deepeval metrics for MediGenius contract tests."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


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


def _normalize_eval_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.replace("℃", "度").replace("°c", "度")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _coerce_claim_specs(claims: list[Any]) -> list[dict[str, Any]]:
    normalized_claims = []
    for item in claims or []:
        if isinstance(item, str):
            claim_text = item.strip()
            match_any = [claim_text] if claim_text else []
            match_all = []
            match_none = []
        elif isinstance(item, dict):
            claim_text = str(
                item.get("claim") or item.get("statement") or item.get("name") or ""
            ).strip()
            raw_match_any = item.get("match_any") or []
            if not raw_match_any and claim_text:
                raw_match_any = [claim_text]
            match_any = [str(term).strip() for term in raw_match_any if str(term).strip()]
            match_all = [
                str(term).strip()
                for term in (item.get("match_all") or [])
                if str(term).strip()
            ]
            match_none = [
                str(term).strip()
                for term in (item.get("match_none") or [])
                if str(term).strip()
            ]
        else:
            continue

        normalized_claims.append(
            {
                "claim": claim_text or "unnamed-claim",
                "match_any": [_normalize_eval_text(term) for term in match_any],
                "match_all": [_normalize_eval_text(term) for term in match_all],
                "match_none": [_normalize_eval_text(term) for term in match_none],
            }
        )
    return normalized_claims


def _claim_matches(normalized_text: str, claim_spec: dict[str, Any]) -> bool:
    match_any = claim_spec.get("match_any") or []
    match_all = claim_spec.get("match_all") or []
    match_none = claim_spec.get("match_none") or []
    any_hit = True if not match_any else any(term in normalized_text for term in match_any)
    all_hit = all(term in normalized_text for term in match_all)
    none_hit = any(term in normalized_text for term in match_none)
    return any_hit and all_hit and not none_hit


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


class ClaimFaithfulnessMetric(BaseMetric):
    """Check whether answer claims stay within evidence-backed claim envelopes."""

    def __init__(
        self,
        supported_claims: list[Any],
        *,
        forbidden_claims: list[Any] | None = None,
        min_supported_claims: int | None = None,
        min_coverage: float = 0.75,
        max_forbidden_claims: int = 0,
    ):
        self.supported_claims = _coerce_claim_specs(supported_claims)
        self.forbidden_claims = _coerce_claim_specs(forbidden_claims or [])
        self.min_supported_claims = (
            len(self.supported_claims)
            if min_supported_claims is None
            else min_supported_claims
        )
        self.min_coverage = min_coverage
        self.max_forbidden_claims = max_forbidden_claims
        self.threshold = min_coverage
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        normalized_text = _normalize_eval_text(test_case.actual_output or "")
        if not normalized_text:
            self.score = 0.0
            self.reason = "actual_output is empty"
            self.success = False
            return self.score

        supported_hits = [
            spec["claim"]
            for spec in self.supported_claims
            if _claim_matches(normalized_text, spec)
        ]
        missing_supported = [
            spec["claim"]
            for spec in self.supported_claims
            if spec["claim"] not in supported_hits
        ]
        forbidden_hits = [
            spec["claim"]
            for spec in self.forbidden_claims
            if _claim_matches(normalized_text, spec)
        ]

        supported_total = len(self.supported_claims)
        coverage = len(supported_hits) / supported_total if supported_total else 1.0
        precision_denominator = len(supported_hits) + len(forbidden_hits)
        claim_precision = (
            len(supported_hits) / precision_denominator
            if precision_denominator
            else 0.0
        )

        self.score = round((coverage + claim_precision) / 2, 4)
        self.reason = (
            f"supported_hits={supported_hits}; missing_supported={missing_supported}; "
            f"forbidden_hits={forbidden_hits}; coverage={coverage:.2f}; "
            f"claim_precision={claim_precision:.2f}"
        )
        self.success = (
            len(supported_hits) >= self.min_supported_claims
            and coverage >= self.min_coverage
            and len(forbidden_hits) <= self.max_forbidden_claims
        )
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
        return "ClaimFaithfulnessMetric"


class AnswerEvidenceCoverageMetric(BaseMetric):
    """Check how many evidence-backed claims are reflected in the answer."""

    def __init__(self, supported_claims: list[Any], threshold: float = 0.75):
        self.supported_claims = _coerce_claim_specs(supported_claims)
        self.threshold = threshold
        self.async_mode = False
        self.strict_mode = True
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        normalized_text = _normalize_eval_text(test_case.actual_output or "")
        if not normalized_text:
            self.score = 0.0
            self.reason = "actual_output is empty"
            self.success = False
            return self.score

        hits = [
            spec["claim"]
            for spec in self.supported_claims
            if _claim_matches(normalized_text, spec)
        ]
        missing = [
            spec["claim"]
            for spec in self.supported_claims
            if spec["claim"] not in hits
        ]

        total = len(self.supported_claims)
        self.score = len(hits) / total if total else 1.0
        self.reason = (
            f"covered {len(hits)}/{total} evidence-backed claims; "
            f"hits={hits}; missing={missing}"
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
        return "AnswerEvidenceCoverageMetric"


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
