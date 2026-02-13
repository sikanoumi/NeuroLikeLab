import os
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("LORA_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_DIR = os.getenv("LORA_ADAPTER_DIR", "/home/sikanoumi/LLaMA-Factory/outputs/yomi_lora_v2_json")
PERSONA_ID = os.getenv("PERSONA_ID", "yomi_proxy_lora_v2")

IN_FILE = Path("experiments") / "eval_20cases.jsonl"
INDEX_FILE = Path("runs") / "index.jsonl"

def extract_json(s: str) -> str:
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        return s[a:b + 1].strip()
    return s.strip()

def build_prompt(turn_id: str, text: str, emotion_json: str) -> str:
    # format事故を避けて f-stringのみ
    return f"""あなたは「僕（ヨミ）」の代理人格です。
出力は必ずJSONのみ。最初の文字は{{、最後の文字は}}。
JSON以外の文章、前置き、コードブロックは禁止。

decisionは defer|propose|ask_clarify のいずれか。
reply と reason_one_line は日本語。

以下のJSONスキーマに従って値を埋めてください:
{{
  "turn_id": "{turn_id}",
  "persona_id": "{PERSONA_ID}",
  "decision": "defer|propose|ask_clarify",
  "confidence": 0.0,
  "reply": "...",
  "reason_one_line": "..."
}}

[入力テキスト]
{text}

[感情(JSON)]
{emotion_json}
"""

def gen_json(model, tok, prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[-1]  # 入力長（生成部分だけ切り出すため）

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    gen_ids = out[0][in_len:]  # 生成部分だけ
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return extract_json(text)

def main():
    ts = int(time.time())
    out_dir = Path("runs") / f"run_eval_lora_json_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "results.jsonl"
    misses_jsonl = out_dir / "misses.jsonl"
    metrics_json = out_dir / "metrics.json"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    total = 0
    hit = 0
    miss = 0
    by_decision = {"defer": 0, "propose": 0, "ask_clarify": 0, "other": 0}

    with (
        IN_FILE.open("r", encoding="utf-8") as f_in,
        out_jsonl.open("w", encoding="utf-8") as f_out,
        misses_jsonl.open("w", encoding="utf-8") as f_miss,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            total += 1

            turn_id = f"lora_{total:04d}"
            emotion_json = json.dumps(case.get("emotion") or {}, ensure_ascii=False)
            prompt = build_prompt(turn_id, case["text"], emotion_json)

            raw = gen_json(model, tok, prompt)

            try:
                persona = json.loads(raw)
            except Exception:
                persona = {"decision": "other", "confidence": 0.0, "reply": "", "reason_one_line": "json_parse_failed"}

            decision = persona.get("decision", "other")
            if decision not in by_decision:
                decision = "other"
            by_decision[decision] += 1

            expected = (case.get("expect") or {}).get("decision")
            ok = (decision == expected)
            if ok:
                hit += 1
            else:
                miss += 1

            rec = {
                "case_id": case.get("case_id"),
                "expected_decision": expected,
                "got_decision": persona.get("decision"),
                "confidence": persona.get("confidence"),
                "persona_reply": persona.get("reply"),
                "persona_reason": persona.get("reason_one_line"),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not ok:
                f_miss.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "total": total,
        "hit": hit,
        "miss": miss,
        "accuracy_decision": (hit / total) if total else 0.0,
        "by_decision": by_decision,
        "type": "eval_lora_json",
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "persona_id": PERSONA_ID,
        "dataset": IN_FILE.name,
        "run_dir": str(out_dir).replace("\\", "/"),
        "notes": "LoRA outputs JSON; decision evaluated by JSON parse",
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    index_record = {
        "run_id": out_dir.name,
        "ts": ts,
        "type": "eval_lora_json",
        "model": f"{BASE_MODEL}+LoRA(JSON)",
        "persona_id": PERSONA_ID,
        "dataset": IN_FILE.name,
        "accuracy": metrics["accuracy_decision"],
        "by_decision": by_decision,
        "miss": miss,
        "run_dir": metrics["run_dir"],
        "notes": "after_lora_json",
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f_idx:
        f_idx.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    print("Saved:", out_dir)
    print("Indexed:", INDEX_FILE)

if __name__ == "__main__":
    main()
