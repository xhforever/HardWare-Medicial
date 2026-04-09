"""Retrieval and reranking capability tests using pytest + deepeval."""

import os
import sys
from unittest.mock import MagicMock, patch

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.reranker import RerankerAgent  # noqa: E402
from app.agents.retriever import RetrieverAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from tests.deepeval_metrics import (  # noqa: E402
    JsonListContainsTermsMetric,
    JsonListTopFieldMetric,
    serialize_payload,
)


def test_retriever_contract_returns_relevant_deduped_chunks():
    state = initialize_conversation_state()
    state["question"] = "贫血 头晕 乏力"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["selected_department"] = "hematology"
    state["selected_department_forced"] = True
    state["retrieval_query"] = "贫血 头晕 乏力"
    state["department_queries"] = {"hematology": "贫血 头晕 乏力"}

    repeated_doc = Document(
        page_content="贫血常见症状包括头晕和乏力，需要结合血常规判断。" * 4,
        metadata={"department": "hematology", "source": "hema.pdf", "page": 1},
    )

    with patch("app.agents.retriever.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [repeated_doc, repeated_doc]
        mock_get_retriever.return_value = mock_retriever
        result = RetrieverAgent(state)

    assert len(result["documents"]) == 1

    case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(result["rag_context"]),
    )
    assert_test(
        case,
        metrics=[
            JsonListTopFieldMetric("scope", "hematology"),
            JsonListContainsTermsMetric("content", ["贫血", "头晕"], min_matches=1),
        ],
        run_async=False,
    )


def test_reranker_contract_promotes_primary_scope_evidence():
    state = initialize_conversation_state()
    state["question"] = "贫血会不会导致头晕"
    state["retrieval_query"] = "贫血 头晕"
    state["primary_department"] = "hematology"
    state["retrieval_scopes"] = ["hematology", "general_medical"]
    state["merged_rag_context"] = [
        {
            "content": "胃肠不适有时也会导致一般性不舒服。",
            "metadata": {"department": "general_medical"},
            "scope": "general_medical",
            "raw_rank": 0,
        },
        {
            "content": "贫血常见症状包括头晕、乏力和活动后心悸。",
            "metadata": {"department": "hematology"},
            "scope": "hematology",
            "raw_rank": 1,
        },
    ]

    result = RerankerAgent(state)

    case = LLMTestCase(
        input=state["question"],
        actual_output=serialize_payload(result["rag_context"]),
    )
    assert_test(
        case,
        metrics=[
            JsonListTopFieldMetric("scope", "hematology"),
            JsonListContainsTermsMetric("content", ["贫血", "头晕"], min_matches=1),
        ],
        run_async=False,
    )

    assert result["rag_context"][0]["rerank_score"] >= result["rag_context"][1]["rerank_score"]
