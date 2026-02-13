import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "experiments" / "eval_100cases.clean.jsonl"
DST = BASE_DIR / "experiments" / "eval_100cases.router.jsonl"

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

def expected_persona_id(emotion: Dict[str, Any], task: str = "default") -> str:
    # creative は task でのみ割り当て（eval_100 は基本 default 想定）
    t = (task or "default").lower().strip()
    if t in {"brainstorm", "ideation", "creative"}:
        return "creative_v0"

    anxiety = float((emotion or {}).get("anxiety", 0.0))
    fatigue = float((emotion or {}).get("fatigue", 0.0))

    # proxy: 高不安 or 高疲労は safety、それ以外 action
    if anxiety >= 0.7 or fatigue >= 0.8:
        return "safety_v0"
    return "action_v0"

def main():
    rows = load_jsonl(SRC)
    out = []
    for r in rows:
        emo = r.get("emotion") or {}
        task = r.get("task", "default")
        r["expected_persona_id"] = expected_persona_id(emo, task=task)
        out.append(r)

    dump_jsonl(DST, out)
    print("Wrote:", DST)
    print("lines:", len(out))

if __name__ == "__main__":
    main()
