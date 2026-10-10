"""Execution logging for observability (tool calls, timing, errors)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent_runs.jsonl"


def log_event(event: dict[str, Any]) -> None:
    """Append one JSON line to the run log."""
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **event}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
