import json
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

# ===== inputs =====
EVAL_PATH = BASE_DIR / "experiments" / "eval_100cases.clean.jsonl"  # evalを流用
API_BASE = "http://127.0.0.1:8011"  # FastAPI /persona を教師として呼ぶ

# ===== outputs =====
OUT_PATH = BASE_DIR / "datasets" / "lora_teacher_100.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===== prompt sources (same as runtime prompts) =====
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_p0.txt"
PERSONA_PROMPT_PATH = BASE_DIR / "prompts" / "persona" / "yomi_proxy_v0.txt"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    BOMあり/なし両対応でJSONLを読む（utf-8-sig）。
    """
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    cases = load_jsonl(EVAL_PATH)

    system_txt = read_text(SYSTEM_PROMPT_PATH).strip()
    persona_txt = read_text(PERSONA_PROMPT_PATH).strip()

    # LLaMA-Factoryで扱いやすいように instruction は固定文字列にする
    instruction = "\n\n".join([system_txt, persona_txt]).strip()

    n_ok = 0
    n_fail = 0
    started = time.time()

    with httpx.Client(timeout=180.0) as client, OUT_PATH.open("w", encoding="utf-8") as f:
        for c in cases:
            text = c.get("text", "")
            emotion = c.get("emotion") or {"joy": 1.0}

            payload = {"text": text, "emotion": emotion, "persona_id": "yomi_proxy_v0"}

            try:
                r = client.post(f"{API_BASE}/persona", json=payload)
                r.raise_for_status()
                data = r.json()
            except Exception:
                n_fail += 1
                continue

            persona = data.get("persona") or {}
            if not isinstance(persona, dict):
                n_fail += 1
                continue

            # output は「JSON文字列」で固定（LoRAにJSON出力を学習させる）
            out_obj = {
                "decision": persona.get("decision", "defer"),
                "reply": persona.get("reply", ""),
                "reason_one_line": persona.get("reason_one_line", ""),
                "memory_actions": persona.get("memory_actions", []),
            }

            sample = {
                "instruction": instruction,
                "input": json.dumps({"text": text, "emotion": emotion}, ensure_ascii=False),
                "output": json.dumps(out_obj, ensure_ascii=False),
                "meta": {
                    "case_id": c.get("case_id"),
                    "expected_decision": c.get("expected_decision"),
                },
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1

    elapsed = time.time() - started
    print("Wrote:", OUT_PATH)
    print("ok:", n_ok, "fail:", n_fail, "elapsed_sec:", round(elapsed, 2))


if __name__ == "__main__":
    main()
