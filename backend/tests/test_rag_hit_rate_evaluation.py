"""Dataset-based retrieval hit-rate evaluation for the medical RAG chain."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.medical_router import MedicalRouterAgent  # noqa: E402
from app.agents.query_rewriter import QueryRewriterAgent  # noqa: E402
from app.agents.reranker import RerankerAgent  # noqa: E402
from app.agents.retriever import RetrieverAgent  # noqa: E402
from app.core.medical_taxonomy import extract_query_terms  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.services.cache_service import cache_service  # noqa: E402
import app.tools.vector_store as vector_store_module  # noqa: E402
from app.tools.vector_store import get_or_create_vectorstore  # noqa: E402


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "eval_cases"
    / "query_rewriter_gold_cases.json"
)
EVAL_CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

MIN_FORCED_RAG_HIT_RATE = float(os.getenv("MIN_FORCED_RAG_HIT_RATE", "0.75"))
MIN_ACTUAL_CHAIN_HIT_RATE = float(os.getenv("MIN_ACTUAL_CHAIN_HIT_RATE", "0.60"))
TOP_K = 3


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _source_basename(case_data: dict) -> str:
    return os.path.basename(case_data.get("source_file") or "")


def _case_terms(case_data: dict) -> list[str]:
    joined = "\n".join(
        [
            case_data.get("source_topic", ""),
            case_data.get("source_excerpt", ""),
            "\n".join(case_data.get("must_keep_terms", []) or []),
            "\n".join(case_data.get("required_department_query_terms", []) or []),
            "\n".join(case_data.get("gold_retrieval_queries", []) or []),
        ]
    )
    return extract_query_terms(joined)


def _page_matches(candidate_page, source_page_hint: int | None, tolerance: int = 3) -> bool:
    if candidate_page is None or source_page_hint is None:
        return False
    try:
        page_num = int(candidate_page)
    except (TypeError, ValueError):
        return False
    return (
        abs(page_num - int(source_page_hint)) <= tolerance
        or abs((page_num + 1) - int(source_page_hint)) <= tolerance
    )


def _chunk_matches_case(chunk: dict, case_data: dict) -> bool:
    metadata = chunk.get("metadata") or {}
    chunk_scope = chunk.get("scope") or metadata.get("department") or metadata.get("domain")
    expected_scope = case_data["expected_scope"]
    if chunk_scope != expected_scope:
        return False

    chunk_source = metadata.get("source_path") or metadata.get("source") or ""
    if _source_basename(case_data) and os.path.basename(str(chunk_source)) != _source_basename(case_data):
        return False

    content = _normalize_text(chunk.get("content", ""))
    case_terms = _case_terms(case_data)
    must_keep_terms = [_normalize_text(term) for term in case_data.get("must_keep_terms", [])]
    required_terms = [_normalize_text(term) for term in case_data.get("required_department_query_terms", [])]
    source_topic = _normalize_text(case_data.get("source_topic", ""))
    source_topic_terms = [_normalize_text(term) for term in extract_query_terms(case_data.get("source_topic", ""))]

    must_hits = sum(1 for term in must_keep_terms if term and term in content)
    required_hits = sum(1 for term in required_terms if term and term in content)
    keyword_hits = sum(1 for term in case_terms if term and term in content)
    source_topic_hits = sum(1 for term in source_topic_terms if term and term in content)
    page_hit = _page_matches(metadata.get("page"), case_data.get("source_page_hint"))

    return (
        page_hit
        or must_hits >= 2
        or required_hits >= 2
        or keyword_hits >= 3
        or (source_topic and source_topic in content)
        or source_topic_hits >= max(1, min(2, len(source_topic_terms)))
    )


def _topk_hit(chunks: list[dict], case_data: dict, top_k: int = TOP_K) -> bool:
    return any(_chunk_matches_case(chunk, case_data) for chunk in (chunks or [])[:top_k])


def _build_base_state(question: str) -> dict:
    state = initialize_conversation_state()
    state["question"] = question
    state["domain"] = "medical"
    state["use_rag"] = True
    return state


def _run_forced_scope_pipeline(case_data: dict) -> dict:
    state = _build_base_state(case_data["user_query"])
    scope = case_data["expected_scope"]
    state["selected_department"] = scope
    state["selected_department_forced"] = True
    state["primary_department"] = scope
    state["department_candidates"] = [{"name": scope, "score": 1.0}]

    with patch("app.agents.query_rewriter.QUERY_REWRITER_USE_LLM", False), patch(
        "app.agents.query_rewriter.get_light_llm", return_value=None
    ):
        state = QueryRewriterAgent(state)
    state = RetrieverAgent(state)
    state = RerankerAgent(state)
    return state


def _run_actual_chain(case_data: dict) -> dict:
    state = _build_base_state(case_data["user_query"])

    with patch("app.agents.medical_router.get_light_llm", return_value=None), patch(
        "app.agents.query_rewriter.QUERY_REWRITER_USE_LLM", False
    ), patch("app.agents.query_rewriter.get_light_llm", return_value=None):
        state = MedicalRouterAgent(state)
        state = QueryRewriterAgent(state)
    state = RetrieverAgent(state)
    state = RerankerAgent(state)
    return state


def _evaluate_cases(pipeline_runner, include_route_stats: bool = False) -> dict:
    hits = 0
    top1_hits = 0
    route_hits = 0
    details = []

    for case_data in EVAL_CASES:
        cache_service.clear()
        state = pipeline_runner(case_data)
        rag_context = list(state.get("rag_context") or [])
        hit = _topk_hit(rag_context, case_data, top_k=TOP_K)
        top1_hit = _topk_hit(rag_context, case_data, top_k=1)
        route_hit = state.get("primary_department") == case_data["expected_scope"]

        hits += int(hit)
        top1_hits += int(top1_hit)
        route_hits += int(route_hit)
        details.append(
            {
                "case_id": case_data["case_id"],
                "expected_scope": case_data["expected_scope"],
                "predicted_scope": state.get("primary_department"),
                "retrieval_scopes": state.get("retrieval_scopes", []),
                "hit": hit,
                "top1_hit": top1_hit,
                "rewrite_reason": state.get("rewrite_reason"),
                "routing_reason": state.get("routing_reason"),
                "top_chunk_scope": (rag_context[0].get("scope") if rag_context else None),
                "top_chunk_page": (
                    (rag_context[0].get("metadata") or {}).get("page") if rag_context else None
                ),
            }
        )

    summary = {
        "cases": len(EVAL_CASES),
        "hits": hits,
        "hit_rate": round(hits / len(EVAL_CASES), 4),
        "top1_hits": top1_hits,
        "top1_hit_rate": round(top1_hits / len(EVAL_CASES), 4),
        "miss_case_ids": [item["case_id"] for item in details if not item["hit"]],
    }
    if include_route_stats:
        summary["route_hits"] = route_hits
        summary["route_hit_rate"] = round(route_hits / len(EVAL_CASES), 4)
    summary["details"] = details
    return summary


@pytest.fixture(scope="module", autouse=True)
def prepare_vectorstore():
    cache_service.clear()
    vector_store_module._vectorstore = None
    get_or_create_vectorstore()
    yield
    cache_service.clear()


def test_forced_scope_rag_hit_rate():
    summary = _evaluate_cases(_run_forced_scope_pipeline)
    printable = {key: value for key, value in summary.items() if key != "details"}
    print("\nforced_scope_rag_eval =", json.dumps(printable, ensure_ascii=False, indent=2))
    assert summary["hit_rate"] >= MIN_FORCED_RAG_HIT_RATE, json.dumps(
        printable, ensure_ascii=False, indent=2
    )


def test_actual_chain_hit_rate():
    summary = _evaluate_cases(_run_actual_chain, include_route_stats=True)
    printable = {key: value for key, value in summary.items() if key != "details"}
    print("\nactual_chain_eval =", json.dumps(printable, ensure_ascii=False, indent=2))
    assert summary["hit_rate"] >= MIN_ACTUAL_CHAIN_HIT_RATE, json.dumps(
        printable, ensure_ascii=False, indent=2
    )
