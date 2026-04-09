"""
MediGenius — services/__init__.py
Lazy exports for service modules to avoid circular imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DatabaseService",
    "db_service",
    "ChatService",
    "chat_service",
    "ECGMonitorService",
    "ecg_monitor_service",
    "ECGReportService",
    "ecg_report_service",
]


class _ServiceProxy:
    """Forward attribute access and patching to the real singleton instance."""

    def __init__(self, module_name: str, attr_name: str) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_attr_name", attr_name)

    def _target(self) -> Any:
        module = import_module(object.__getattribute__(self, "_module_name"))
        return getattr(module, object.__getattribute__(self, "_attr_name"))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._target(), name)

    def __repr__(self) -> str:
        return repr(self._target())


db_service = _ServiceProxy("app.services.database_service", "db_service")
chat_service = _ServiceProxy("app.services.chat_service", "chat_service")
ecg_monitor_service = _ServiceProxy(
    "app.services.ecg_monitor_service",
    "ecg_monitor_service",
)
ecg_report_service = _ServiceProxy("app.services.ecg_report_service", "ecg_report_service")


def __getattr__(name: str) -> Any:
    module_map = {
        "DatabaseService": "app.services.database_service",
        "ChatService": "app.services.chat_service",
        "ECGMonitorService": "app.services.ecg_monitor_service",
        "ECGReportService": "app.services.ecg_report_service",
    }
    module_name = module_map.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
