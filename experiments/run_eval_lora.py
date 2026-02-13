# experiments/run_eval_lora.py
import os
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("LORA_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_DIR = os.getenv("LORA_ADAPTER_DIR", "/home/sikanoumi/LLaMA-Factory/outputs/yomi_lora_v1_mistral")

IN_FILE = Path("experiments") / "eval_20cases.jsonl"
INDEX_FILE = Path("runs") / "index.jsonl"

def infer_decision(reply: str) -> str:
    """LoRAの自然文出力から、暫定ルールでdecisionを推定（最短でAfter比較を出す）"""
    s = reply.strip()

    # ask_clarify: 質問が主
    if "？" in s or "?" in s or "教えて" in s or "確認" in s:
        return "ask_clarify"

    # defer: 延期/保留/休憩/期限交渉
    defer_kw = ["保留", "延期", "休憩", "明日", "後で", "期限", "再交渉", "今日は決めない", "今すぐは難しい"]
    if any(k in s for k in defer_kw):
        return "defer"

    # propose: それ以外は提案扱い
    return "propose"

def build_prompt(text: str, emotion: dict) -> str:
    # 学習データで見えていた形式に寄せる（Human/Assistant）
    return (
        "Human: あなたはヨミの代理人格です。断定しすぎず、判断は保留でき、感情（anxiety/confidence/fatigue）を織り込んで返答してください。"
        "最後に理由(一行)も付けてください。\n"
        f"text={text}\n"
        f"emotion={json.dumps(emotion, ensure_ascii=False)}\n"
        "Assistant: "
    )

def gen_text(model, tok, prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            temperature=0.0,
        )
    full = tok.decode(out[0], skip_special_tokens=True)

    # "Assistant: " 以降だけ抜く
    idx = full.rfind("Assistant:")
    if idx != -1:
        return full[idx + len("Assistant:"):].strip()
    # fallback
    return full.replace(prompt, "").strip()

def main():
    ts = int(time.time())
    out_dir = Path("runs") / f"run_eval_lora_{ts}"
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

            text = case["text"]
            emotion = case.get("emotion") or {}
            prompt = build_prompt(text, emotion)

            reply = gen_text(model, tok, prompt)
            decision = infer_decision(reply)

            expected = (case.get("expect") or {}).get("decision")
            ok = (decision == expected)
            if ok:
                hit += 1
            else:
                miss += 1

            if decision not in by_decision:
                by_decision["other"] += 1
            else:
                by_decision[decision] += 1

            rec = {
                "case_id": case.get("case_id"),
                "expected_decision": expected,
                "got_decision": decision,
                "reply": reply,
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
        "type": "eval_lora",
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "run_dir": str(out_dir).replace("\\", "/"),
        "notes": "LoRA outputs natural reply; decision inferred by heuristic",
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # index.jsonl にも追記
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    index_record = {
        "run_id": out_dir.name,
        "ts": ts,
        "type": "eval_lora",
        "model": f"{BASE_MODEL}+LoRA",
        "persona_id": "yomi_proxy_lora_v1",
        "dataset": IN_FILE.name,
        "accuracy": metrics["accuracy_decision"],
        "by_decision": by_decision,
        "miss": miss,
        "run_dir": metrics["run_dir"],
        "notes": "after_lora (heuristic decision)",
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f_idx:
        f_idx.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    print("Saved:", out_dir)
    print("Indexed:", INDEX_FILE)

if __name__ == "__main__":
    main()
