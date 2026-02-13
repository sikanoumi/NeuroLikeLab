import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

EVAL_PATH = BASE_DIR / "experiments" / "eval_100cases.clean.jsonl"
API_BASE = "http://127.0.0.1:8011"

OUT_PATH = BASE_DIR / "datasets" / "lora_teacher_100_v2_retry.jsonl"
OUT_FAIL = BASE_DIR / "datasets" / "lora_teacher_100_v2_retry_fail.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_p0.txt"
PERSONA_PROMPT_PATH = BASE_DIR / "prompts" / "persona" / "yomi_proxy_v0.txt"

MAX_TRY = 5  # ←固定


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    cases = load_jsonl(EVAL_PATH)
    instruction = "\n\n".join([read_text(SYSTEM_PROMPT_PATH).strip(), read_text(PERSONA_PROMPT_PATH).strip()]).strip()

    n_ok = 0
    n_fail = 0
    started = time.time()

    with httpx.Client(timeout=180.0) as client, OUT_PATH.open("w", encoding="utf-8") as f_ok, OUT_FAIL.open("w", encoding="utf-8") as f_fail:
        for c in cases:
            text = c.get("text", "")
            emotion = c.get("emotion") or {"joy": 1.0}
            expected = c.get("expected_decision")

            # expected が無いケースはそのまま1回で採用（今は全件ある想定）
            if not expected:
                expected = None

            best_persona: Optional[Dict[str, Any]] = None
            last_err: Optional[str] = None

            for t in range(1, MAX_TRY + 1):
                payload = {"text": text, "emotion": emotion, "persona_id": "yomi_proxy_v0"}
                try:
                    r = client.post(f"{API_BASE}/persona", json=payload)
                    r.raise_for_status()
                    data = r.json()
                except Exception as e:
                    last_err = str(e)
                    continue

                persona = data.get("persona")
                if not isinstance(persona, dict):
                    last_err = "persona_not_dict"
                    continue

                # まず保持
                best_persona = persona

                # expectedがあるなら一致するまでリトライ
                if expected is None or persona.get("decision") == expected:
                    break

            if best_persona is None:
                n_fail += 1
                f_fail.write(json.dumps({
                    "case_id": c.get("case_id"),
                    "expected_decision": expected,
                    "error": last_err,
                    "text": text,
                    "emotion": emotion,
                }, ensure_ascii=False) + "\n")
                continue

            # output（JSON文字列）を固定
            out_obj = {
                "decision": best_persona.get("decision", "defer"),
                "reply": best_persona.get("reply", ""),
                "reason_one_line": best_persona.get("reason_one_line", ""),
                "memory_actions": best_persona.get("memory_actions", []),
            }

            # データセット1行
            sample = {
                "instruction": instruction,
                "input": json.dumps({"text": text, "emotion": emotion}, ensure_ascii=False),
                "output": json.dumps(out_obj, ensure_ascii=False),
                "meta": {
                    "case_id": c.get("case_id"),
                    "expected_decision": expected,
                },
            }

            # 一致してない場合もある（MAX_TRY上限）
            if expected is not None and out_obj["decision"] != expected:
                n_fail += 1
                f_fail.write(json.dumps({
                    "case_id": c.get("case_id"),
                    "expected_decision": expected,
                    "got_decision": out_obj["decision"],
                    "text": text,
                    "emotion": emotion,
                }, ensure_ascii=False) + "\n")
                continue

            f_ok.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1

    elapsed = time.time() - started
    print("Wrote:", OUT_PATH)
    print("Wrote(fail):", OUT_FAIL)
    print("ok:", n_ok, "fail:", n_fail, "elapsed_sec:", round(elapsed, 2))


if __name__ == "__main__":
    main()
