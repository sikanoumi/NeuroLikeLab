import json
from pathlib import Path
from typing import Any, Dict

LOGS_DIR = Path(__file__).parent.parent / "logs"


def append_turn_log(turn_data: Dict[str, Any]) -> None:
    """Append a turn record to logs/turns.jsonl (UTF-8, ensure_ascii=False)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "turns.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")
