"""
MediGenius — core/config.py
Environment variables and path constants.
"""

import os

from dotenv import load_dotenv

from app.core.medical_taxonomy import department_folder_name, list_department_codes

load_dotenv()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ── Paths ──────────────────────────────────────────────────────────────────────
# backend/app/core/config.py -> backend/
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Ensure logs and storage are inside backend directory
LOG_DIR = os.getenv("LOG_DIR", os.path.join(_BACKEND_DIR, "logs"))
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(_BACKEND_DIR, "storage", "chat_db", "medigenius.db"))
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(_BACKEND_DIR, "storage", "vector_store"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(_BACKEND_DIR, "data", "medical_book.pdf"))
KNOWLEDGE_ROOT_DIR = os.getenv(
    "KNOWLEDGE_ROOT_DIR",
    os.path.join(_BACKEND_DIR, "data", "knowledge"),
)
DEPARTMENT_KNOWLEDGE_DIR = os.getenv(
    "DEPARTMENT_KNOWLEDGE_DIR",
    os.path.join(KNOWLEDGE_ROOT_DIR, "departments"),
)
GENERAL_MEDICAL_KNOWLEDGE_DIR = os.getenv(
    "GENERAL_MEDICAL_KNOWLEDGE_DIR",
    os.path.join(KNOWLEDGE_ROOT_DIR, "medical", department_folder_name("general_medical")),
)
PROFILE_STORE_DIR = os.getenv("PROFILE_STORE_DIR", os.path.join(_BACKEND_DIR, "storage", "profiles"))
ECG_REPORT_PDF_DIR = os.getenv(
    "ECG_REPORT_PDF_DIR",
    os.path.join(_BACKEND_DIR, "storage", "ecg_reports"),
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_ENABLED = _env_bool("RAG_ENABLED", True)
WEB_SEARCH_ENABLED = _env_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_USE_LLM_DECIDER = _env_bool("WEB_SEARCH_USE_LLM_DECIDER", False)
QUERY_REWRITER_ENABLED = _env_bool("QUERY_REWRITER_ENABLED", True)
QUERY_REWRITER_USE_LLM = _env_bool("QUERY_REWRITER_USE_LLM", True)
TOKEN_BUDGET_ENABLED = _env_bool("TOKEN_BUDGET_ENABLED", True)
SESSION_SUMMARY_ENABLED = _env_bool("SESSION_SUMMARY_ENABLED", True)
SESSION_SUMMARY_USE_LLM = _env_bool("SESSION_SUMMARY_USE_LLM", False)
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
HISTORY_BOOTSTRAP_LIMIT = int(os.getenv("HISTORY_BOOTSTRAP_LIMIT", "40"))
HISTORY_HARD_LIMIT = int(os.getenv("HISTORY_HARD_LIMIT", "20"))
EXECUTOR_PROMPT_MAX_TOKENS = int(os.getenv("EXECUTOR_PROMPT_MAX_TOKENS", "2600"))
EXECUTOR_COMPLETION_RESERVE_TOKENS = int(
    os.getenv("EXECUTOR_COMPLETION_RESERVE_TOKENS", "600")
)
TOKEN_BUDGET_PROFILE_TOKENS = int(os.getenv("TOKEN_BUDGET_PROFILE_TOKENS", "180"))
TOKEN_BUDGET_SUMMARY_TOKENS = int(os.getenv("TOKEN_BUDGET_SUMMARY_TOKENS", "220"))
TOKEN_BUDGET_RECENT_HISTORY_TOKENS = int(
    os.getenv("TOKEN_BUDGET_RECENT_HISTORY_TOKENS", "360")
)
TOKEN_BUDGET_RAG_TOKENS = int(os.getenv("TOKEN_BUDGET_RAG_TOKENS", "900"))
TOKEN_BUDGET_WEB_TOKENS = int(os.getenv("TOKEN_BUDGET_WEB_TOKENS", "320"))
TOKEN_BUDGET_MIN_SECTION_TOKENS = int(
    os.getenv("TOKEN_BUDGET_MIN_SECTION_TOKENS", "48")
)

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_WIRE_API = os.getenv("OPENAI_WIRE_API", "chat").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-plus")
LIGHT_LLM_MODEL = os.getenv("LIGHT_LLM_MODEL", "qwen3.5-flash")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_ROUTING_CONFIG_PATH = os.getenv(
    "MODEL_ROUTING_CONFIG_PATH",
    os.path.join(_BACKEND_DIR, "storage", "model_routing.json"),
)

# ── ECG Remote Monitor ────────────────────────────────────────────────────────
ECG_SITE_URL = os.getenv(
    "ECG_SITE_URL",
    "http://124.220.204.12:8080/index#/system/doctor",
)
ECG_SITE_USER = os.getenv("ECG_SITE_USER", "doctor")
ECG_SITE_PASS = os.getenv("ECG_SITE_PASS", "123456")

DEPARTMENT_HINT_FOLDERS = {
    code: department_folder_name(code)
    for code in list_department_codes()
}
