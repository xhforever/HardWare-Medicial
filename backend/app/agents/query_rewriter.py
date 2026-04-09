"""
MediGenius — agents/query_rewriter.py
Rewrite the user question into retrieval-focused queries.
"""

from __future__ import annotations

import json

from app.core.config import QUERY_REWRITER_ENABLED, QUERY_REWRITER_USE_LLM
from app.core.logging_config import logger
from app.core.medical_taxonomy import (
    GENERAL_MEDICAL_DEPARTMENT,
    department_display_name,
    extract_query_terms,
)
from app.core.state import AgentState, append_flow_trace
from app.services.cache_service import cache_service, record_cache_result
from app.tools.llm_client import (
    coerce_response_text,
    get_light_llm,
    invoke_with_metrics,
)

MEDICAL_QUERY_EXPANSIONS = {
    "偏头痛": ["migraine"],
    "糖尿病": ["diabetes"],
    "哮喘": ["asthma"],
    "肺炎": ["pneumonia"],
    "重症肺炎": ["severe pneumonia"],
    "腹泻": ["diarrhoea", "diarrhea"],
    "急性腹泻": ["acute diarrhoea", "acute diarrhea"],
    "腹泻性疾病": ["diarrheal diseases"],
    "麻疹": ["measles"],
    "乳突炎": ["mastoiditis"],
    "中耳炎": ["otitis media"],
    "脑膜炎": ["meningitis"],
    "泌尿道感染": ["urinary tract infection", "uti"],
    "尿路感染": ["urinary tract infection", "uti"],
    "丙型肝炎": ["hepatitis c", "hcv"],
    "丙肝": ["hepatitis c", "hcv"],
    "乙肝": ["hepatitis b", "hbv"],
    "疟疾": ["malaria"],
    "结核": ["tuberculosis"],
    "艾滋": ["hiv"],
    "hiv": ["aids"],
    "性传播感染": ["sexually transmitted infection", "sti"],
    "性传播": ["sexually transmitted infection", "sti"],
}


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _expand_cross_lingual_terms(question: str, terms: list[str]) -> list[str]:
    expanded_terms = list(terms)
    seen = set(expanded_terms)
    lowered_question = (question or "").lower()

    for trigger, expansions in MEDICAL_QUERY_EXPANSIONS.items():
        if trigger.lower() not in lowered_question:
            continue
        for expansion in expansions:
            for token in extract_query_terms(expansion):
                if token in seen:
                    continue
                seen.add(token)
                expanded_terms.append(token)
    return expanded_terms


def _fallback_retrieval_query(question: str, scope: str | None = None) -> str:
    terms = _expand_cross_lingual_terms(question, extract_query_terms(question))
    if scope and scope != GENERAL_MEDICAL_DEPARTMENT:
        terms.insert(0, department_display_name(scope))
    if not terms:
        return question.strip()
    return " ".join(terms[:16])


def QueryRewriterAgent(state: AgentState) -> AgentState:
    """Generate retrieval-oriented queries while preserving the original question."""
    append_flow_trace(state, "query_rewriter")
    question = (state.get("question") or "").strip()
    if not question:
        return state

    domain = state.get("domain", "general")
    forced_department_mode = bool(
        state.get("selected_department_forced") and state.get("selected_department")
    )
    if forced_department_mode:
        scopes = [state["selected_department"]]
    else:
        candidate_departments = [
            item.get("name")
            for item in state.get("department_candidates", [])
            if isinstance(item, dict) and item.get("name")
        ]
        if domain == "medical":
            scopes = candidate_departments[:3] or [state.get("primary_department") or GENERAL_MEDICAL_DEPARTMENT]
        elif state.get("use_rag") or state.get("need_rag"):
            scopes = [domain]
        else:
            scopes = []

    fallback_query = _fallback_retrieval_query(question)
    fallback_department_queries = {
        scope: _fallback_retrieval_query(question, scope)
        for scope in scopes
    }
    rewrite_reason = "heuristic keyword normalization"
    retrieval_query = fallback_query
    department_queries = fallback_department_queries

    if not QUERY_REWRITER_ENABLED:
        state["retrieval_query"] = question or fallback_query
        state["department_queries"] = {
            scope: state["retrieval_query"] for scope in scopes
        }
        state["rewrite_reason"] = "query rewriter disabled by config"
        logger.info(
            "QueryRewriter: disabled by config, retrieval_query=%s scopes=%s",
            state["retrieval_query"][:80],
            list(state["department_queries"].keys()),
        )
        return state

    if forced_department_mode:
        # Latency-first path: manual department selection already disambiguates scope.
        state["retrieval_query"] = fallback_department_queries.get(scopes[0], fallback_query)
        state["department_queries"] = fallback_department_queries
        state["rewrite_reason"] = "manual department fast-path"
        logger.info(
            "QueryRewriter: retrieval_query=%s scopes=%s",
            state["retrieval_query"][:80],
            list(state["department_queries"].keys()),
        )
        return state

    if not QUERY_REWRITER_USE_LLM:
        state["retrieval_query"] = fallback_query
        state["department_queries"] = fallback_department_queries
        state["rewrite_reason"] = "heuristic keyword normalization (llm disabled)"
        logger.info(
            "QueryRewriter: llm disabled, retrieval_query=%s scopes=%s",
            state["retrieval_query"][:80],
            list(state["department_queries"].keys()),
        )
        return state

    llm = get_light_llm(
        tenant_id=state.get("tenant_id", "default"),
        user_id=state.get("user_id", "anonymous"),
    )
    cache_key = cache_service.make_key(
        "query_rewriter",
        {
            "tenant_id": state.get("tenant_id", "default"),
            "user_id": state.get("user_id", "anonymous"),
            "question": question,
            "domain": domain,
            "scopes": scopes,
            "candidate_departments": candidate_departments,
        },
    )
    cached = cache_service.get(cache_key)
    if cached:
        state["retrieval_query"] = cached.get("retrieval_query", fallback_query)
        state["department_queries"] = cached.get("department_queries", fallback_department_queries)
        state["rewrite_reason"] = cached.get("rewrite_reason", rewrite_reason)
        record_cache_result(state, "query_rewriter", True)
        logger.info(
            "QueryRewriter: cache hit, retrieval_query=%s scopes=%s",
            state["retrieval_query"][:80],
            list(state["department_queries"].keys()),
        )
        return state
    record_cache_result(state, "query_rewriter", False)

    if llm and scopes:
        prompt = (
            "你负责把用户问题改写成更适合医疗检索的查询语句。\n"
            "只返回如下 JSON：\n"
            "{"
            "\"retrieval_query\": \"...\", "
            "\"department_queries\": {\"scope\": \"...\"}, "
            "\"rewrite_reason\": \"...\""
            "}\n"
            "查询语句必须是简洁的检索短语，不能直接回答用户问题。\n"
            f"检索范围：{', '.join(scopes)}\n"
            f"用户问题：{question[:1200]}\n"
        )
        try:
            raw = invoke_with_metrics(
                llm,
                prompt,
                node_name="query_rewriter",
                state=state,
            )
            content = coerce_response_text(raw)
            parsed = json.loads(_extract_json_block(content))
            retrieval_query = (
                str(parsed.get("retrieval_query") or fallback_query).strip() or fallback_query
            )
            raw_department_queries = parsed.get("department_queries") or {}
            for scope in scopes:
                candidate = raw_department_queries.get(scope)
                if candidate:
                    department_queries[scope] = str(candidate).strip()
            rewrite_reason = str(parsed.get("rewrite_reason") or rewrite_reason)
            cache_service.set(
                cache_key,
                {
                    "retrieval_query": retrieval_query,
                    "department_queries": department_queries,
                    "rewrite_reason": rewrite_reason,
                },
            )
        except Exception as exc:
            logger.warning("QueryRewriter fallback used: %s", exc)

    state["retrieval_query"] = retrieval_query
    state["department_queries"] = department_queries
    state["rewrite_reason"] = rewrite_reason
    logger.info(
        "QueryRewriter: retrieval_query=%s scopes=%s",
        retrieval_query[:80],
        list(department_queries.keys()),
    )
    return state
