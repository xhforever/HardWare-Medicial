"""
MediGenius — tools/vector_store.py
ChromaDB vector store: embeddings, creation, loading, and retriever factory.
"""

import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from app.core.config import EMBEDDING_MODEL_NAME, KNOWLEDGE_ROOT_DIR, VECTOR_STORE_DIR
from app.core.logging_config import logger
from app.core.medical_taxonomy import (
    GENERAL_MEDICAL_DEPARTMENT,
    normalize_department_code,
)

_embeddings = None
_vectorstore = None


def _create_vectorstore_from_documents(Chroma, documents, embeddings, persist_dir: str):
    """Rebuild a persisted Chroma collection from fresh documents."""
    global _vectorstore

    if not documents:
        return None

    try:
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("Creating new vector store with %d documents", len(documents))
        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        _vectorstore.persist()
        return _vectorstore
    except Exception as exc:
        logger.error("Failed to create vector store: %s", exc)
        _vectorstore = None
        return None


def _collection_has_metadata_key(vectorstore, key: str, sample_size: int = 64) -> bool:
    """Check whether existing collection chunks carry a given metadata key."""
    try:
        payload = vectorstore._collection.get(limit=sample_size, include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        return any(isinstance(meta, dict) and key in meta for meta in metadatas)
    except Exception:
        return False


def _discover_knowledge_departments_on_disk(root_dir: str) -> Set[str]:
    """Return departments that currently have at least one knowledge file on disk."""
    knowledge_root = Path(root_dir)
    if not knowledge_root.exists():
        return set()

    departments: Set[str] = set()
    for path in knowledge_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".epub"}:
            continue
        try:
            relative_parts = path.relative_to(knowledge_root).parts[:-1]
        except ValueError:
            relative_parts = path.parts[:-1]

        department = None
        for part in relative_parts:
            department = normalize_department_code(part)
            if department:
                break
        departments.add(department or GENERAL_MEDICAL_DEPARTMENT)
    return departments


def _document_departments(documents: Optional[List[Document]]) -> Set[str]:
    departments: Set[str] = set()
    for doc in documents or []:
        metadata = doc.metadata or {}
        department = normalize_department_code(metadata.get("department"))
        if department:
            departments.add(department)
    return departments


def _collection_has_department(vectorstore, department: str) -> bool:
    try:
        payload = vectorstore._collection.get(
            where={"department": department},
            limit=1,
            include=["metadatas"],
        )
        ids = payload.get("ids") or []
        metadatas = payload.get("metadatas") or []
        return bool(ids or metadatas)
    except Exception:
        return False


def _missing_collection_departments(vectorstore, required_departments: Set[str]) -> List[str]:
    missing = []
    for department in sorted(required_departments):
        if not _collection_has_department(vectorstore, department):
            missing.append(department)
    return missing


def _load_knowledge_library_documents() -> List[Document]:
    from app.tools.pdf_loader import process_knowledge_library

    return process_knowledge_library(KNOWLEDGE_ROOT_DIR)


def _load_department_documents(departments: Set[str]) -> List[Document]:
    from app.tools.pdf_loader import process_knowledge_library

    return process_knowledge_library(KNOWLEDGE_ROOT_DIR, allowed_departments=departments)


def _filter_documents_by_departments(
    documents: Optional[List[Document]],
    departments: Set[str],
) -> List[Document]:
    filtered: List[Document] = []
    for doc in documents or []:
        metadata = doc.metadata or {}
        department = normalize_department_code(metadata.get("department"))
        if department in departments:
            filtered.append(doc)
    return filtered


def _rebuild_vectorstore(
    Chroma,
    embeddings,
    persist_dir: str,
    documents: Optional[List[Document]],
    rebuild_reason: str,
):
    fresh_documents = list(documents or [])
    if not fresh_documents:
        fresh_documents = _load_knowledge_library_documents()
    if not fresh_documents:
        logger.warning(
            "Vector store rebuild skipped: %s, and no fresh knowledge documents found.",
            rebuild_reason,
        )
        return None

    logger.warning("%s Rebuilding vector store from fresh knowledge documents.", rebuild_reason)
    shutil.rmtree(persist_dir, ignore_errors=True)
    return _create_vectorstore_from_documents(Chroma, fresh_documents, embeddings, persist_dir)


def _augment_vectorstore_departments(
    vectorstore,
    documents: Optional[List[Document]],
    missing_departments: Set[str],
):
    fresh_documents = _filter_documents_by_departments(documents, missing_departments)
    if not fresh_documents:
        fresh_documents = _load_department_documents(missing_departments)
    if not fresh_documents:
        logger.warning(
            "Vector store department repair skipped: no fresh documents found for %s",
            sorted(missing_departments),
        )
        return None

    logger.warning(
        "Adding %d document chunks for missing departments: %s",
        len(fresh_documents),
        sorted(missing_departments),
    )
    try:
        vectorstore.add_documents(fresh_documents)
        vectorstore.persist()
        return vectorstore
    except Exception as exc:
        logger.error(
            "Failed to augment vector store for missing departments %s: %s",
            sorted(missing_departments),
            exc,
        )
        return None


def _ensure_department_coverage(
    vectorstore,
    Chroma,
    embeddings,
    persist_dir: str,
    documents: Optional[List[Document]],
):
    required_departments = _document_departments(documents)
    if not required_departments:
        required_departments = _discover_knowledge_departments_on_disk(KNOWLEDGE_ROOT_DIR)
    if not required_departments:
        return vectorstore

    if not _collection_has_metadata_key(vectorstore, "department"):
        return _rebuild_vectorstore(
            Chroma,
            embeddings,
            persist_dir,
            documents,
            "Existing vector store lacks `department` metadata.",
        )

    missing_departments = _missing_collection_departments(vectorstore, required_departments)
    if missing_departments:
        repaired_vectorstore = _augment_vectorstore_departments(
            vectorstore,
            documents,
            set(missing_departments),
        )
        if repaired_vectorstore:
            return repaired_vectorstore

        return _rebuild_vectorstore(
            Chroma,
            embeddings,
            persist_dir,
            documents,
            "Existing vector store is missing department coverage: "
            + ", ".join(missing_departments)
            + ".",
        )

    return vectorstore


def get_embeddings():
    """Return a cached HuggingFace sentence-transformer embeddings instance."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
            )
            logger.info("Embeddings model loaded (%s) on CPU", EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.error("Failed to initialize embeddings model: %s", exc)
            return None
    return _embeddings


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    persist_dir: str = VECTOR_STORE_DIR,
):
    """Load existing ChromaDB vector store or create a new one from documents."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings()
    if embeddings is None:
        logger.warning("Vector store unavailable: embeddings model not ready")
        return None

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db_files_exist = any(
        f.endswith(".sqlite3") or f == "chroma.sqlite3" or f.startswith("index")
        for f in os.listdir(persist_dir)
    ) if os.path.exists(persist_dir) else False

    if db_files_exist:
        try:
            logger.info("Loading existing vector store from %s", persist_dir)
            _vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            if _vectorstore._collection.count() == 0:
                logger.warning("Vector store is empty — needs to be recreated")
                _vectorstore = None
                return _rebuild_vectorstore(
                    Chroma,
                    embeddings,
                    persist_dir,
                    documents,
                    "Existing vector store is empty.",
                )

            _vectorstore = _ensure_department_coverage(
                _vectorstore,
                Chroma,
                embeddings,
                persist_dir,
                documents,
            )
            if _vectorstore is None:
                return None

            logger.info(
                "Loaded %d documents from vector store", _vectorstore._collection.count()
            )
        except Exception as exc:
            logger.error("Failed to load vector store: %s", exc)
            _vectorstore = None
            return _rebuild_vectorstore(
                Chroma,
                embeddings,
                persist_dir,
                documents,
                "Existing vector store appears invalid.",
            )
    elif documents:
        return _create_vectorstore_from_documents(Chroma, documents, embeddings, persist_dir)
    else:
        logger.warning("No existing vector store and no documents provided")
        return None

    return _vectorstore


def get_retriever(k: int = 3, search_kwargs: Optional[Dict] = None):
    """Return a retriever from the vector store, or None if unavailable."""
    vs = get_or_create_vectorstore()
    if vs:
        kwargs = {"k": k}
        if search_kwargs:
            kwargs.update(search_kwargs)
        return vs.as_retriever(search_kwargs=kwargs)
    return None
