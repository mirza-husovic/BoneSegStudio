"""Application-wide logging.

One rotating file per install (logs/boneseg.log, 5 MB x 5 backups) plus a
console handler. Every module obtains its logger via ``get_logger(__name__)``
so log lines carry the originating layer (inference, pipeline, ui, ...).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from boneseg.config import LOGS_DIR, ensure_runtime_dirs

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root handlers once; subsequent calls are no-ops."""
    global _configured
    if _configured:
        return

    ensure_runtime_dirs()

    root = logging.getLogger("boneseg")
    root.setLevel(level)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "boneseg.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``boneseg`` hierarchy."""
    setup_logging()
    if not name.startswith("boneseg"):
        name = f"boneseg.{name}"
    return logging.getLogger(name)
