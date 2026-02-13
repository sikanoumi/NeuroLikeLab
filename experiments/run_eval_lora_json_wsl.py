import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Paths (WSL) ---
NEUROLIKE_DIR = Path("/mnt/c/Users/志賀海/開発/NeuroLikeLab")
EVAL_PATH = NEUROLIKE_DIR / "experiments" / "eval_100cases.clean.jsonl"
RUNS_DIR = NEUROLIKE_DIR / "runs"

SYSTEM_PROMPT_PATH = NEUROLIKE_DIR / "prompts" / "system_p0.txt"
PERSONA_PROMPT_PATH = NEUROLIKE_DIR / "prompts" / "persona" / "yomi_proxy_v0.txt"

# LoRA adapter dir (WSL)
LORA_ADAPTER_DIR = Path("/home/sikanoumi/LLaMA-Factory/outputs/yomi_lora_v2_json")


# Base model (same as training)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

ALLOWED_DECISIONS = {"defer", "propose", "ask_clarify"}
ALLOWED_ACTIONS = {"add", "retrieve", "summarize", "update", "none"}
ALLOWED_TARGETS = {"stm", "work", "ltm"}


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def extract_json_object(s: str) -> Tuple[Optional[str], Optional[str]]:
    if not s:
        return None, "empty_response"
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_braces_found"
    return s[start : end + 1].strip(), None


def normalize_target_alias(t: str) -> str:
    t = (t or "stm").lower().strip()
    alias = {
        "work_memo": "work",
        "workmemo": "work",
        "work-memo": "work",
        "memo": "work",
        "recent_stm": "stm",
        "stm_turns": "stm",
        "short_term": "stm",
        "long_term": "ltm",
        "ltm_items": "ltm",
    }
    return alias.get(t, t)


def validate_and_sanitize_persona(persona_obj: dict) -> Tuple[dict, dict]:
    report: Dict[str, Any] = {
        "validated_ok": True,
        "forced_decision": False,
        "trimmed_actions": 0,
        "dropped_actions": [],
        "kept_actions": 0,
    }

    p = dict(persona_obj or {})

    decision = (p.get("decision") or "defer").lower().strip()
    if decision not in ALLOWED_DECISIONS:
        p["decision"] = "defer"
        report["forced_decision"] = True
    else:
        p["decision"] = decision

    p["reply"] = str(p.get("reply") or "")
    p["reason_one_line"] = str(p.get("reason_one_line") or "")

    acts = p.get("memory_actions") or []
    if not isinstance(acts, list):
        acts = []

    sanitized: List[Dict[str, Any]] = []
    for a in acts:
        if not isinstance(a, dict):
            report["dropped_actions"].append({"reason": "not_dict", "item": str(a)[:80]})
            continue

        act = (a.get("action") or "none").lower().strip()
        tgt_raw = a.get("target") or "stm"
        tgt = normalize_target_alias(str(tgt_raw))
        note = (a.get("note") or "").strip()

        if act not in ALLOWED_ACTIONS:
            report["dropped_actions"].append({"reason": "bad_action", "action": act, "target": str(tgt_raw)})
            continue

        if act != "none" and tgt not in ALLOWED_TARGETS:
            report["dropped_actions"].append({"reason": "bad_target", "action": act, "target": str(tgt_raw)})
            continue

        if act in {"add", "update"} and not note:
            report["dropped_actions"].append({"reason": "empty_note", "action": act, "target": tgt})
            continue

        if len(note) > 120:
            note = note[:120]

        sanitized.append({"action": act, "target": tgt, "note": note})

    if len(sanitized) > 4:
        report["trimmed_actions"] = len(sanitized) - 4
        sanitized = sanitized[:4]

    report["kept_actions"] = len(sanitized)
    p["memory_actions"] = sanitized
    return p, report


def build_instruction() -> str:
    system_txt = read_text(SYSTEM_PROMPT_PATH).strip()
    persona_txt = read_text(PERSONA_PROMPT_PATH).strip()
    return "\n\n".join([system_txt, persona_txt]).strip()


def build_prompt(tokenizer, instruction: str, text: str, emotion: Dict[str, Any]) -> str:
    # 学習時と同じく input は JSON 文字列
    user_input = json.dumps({"text": text, "emotion": emotion}, ensure_ascii=False)
    content = instruction + "\n" + user_input

    # chat template があればそれを使う（mistral 系）
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def main():
    # NeuroLikeLab core.memory を使って action_results も算出（WSLからWindowsのコードをimport）
    sys.path.append(str(NEUROLIKE_DIR))
    from core.memory import apply_memory_actions  # type: ignore

    cases = load_jsonl(EVAL_PATH)
    instruction = build_instruction()

    # Load base + LoRA (4bit)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_DIR))
    model.eval()

    total = 0
    ok = 0
    invalid_json = 0

    decision_eval_n = 0
    decision_correct = 0

    total_actions = 0
    dropped_actions = 0
    forced_decision = 0

    mem_pollution = 0  # rejected + skipped_duplicates (簡易)
    retrieve_count = 0
    unnecessary_retrieve = 0

    started = time.time()

    for c in cases:
        total += 1
        text = c.get("text", "")
        emotion = c.get("emotion") or {"joy": 1.0}
        expected = c.get("expected_decision")

        prompt = build_prompt(tokenizer, instruction, text, emotion)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # deterministic generation
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )

        # ★入力分を切り落として「生成された部分だけ」を decode（超重要）
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][input_len:]
        gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # たまに ```json ``` が混ざるので除去
        gen = gen.replace("```json", "").replace("```", "").strip()

        # JSON抽出
        json_str, err = extract_json_object(gen)
        if err:
            invalid_json += 1
            continue

        try:
            persona_obj = json.loads(json_str)
        except Exception:
            invalid_json += 1
            continue

        ok += 1

        # sanitize + obedience-like report
        persona_obj, ob_report = validate_and_sanitize_persona(persona_obj)
        if ob_report.get("forced_decision"):
            forced_decision += 1
        dropped_actions += len(ob_report.get("dropped_actions") or [])

        # decision accuracy (expectedありだけ)
        if expected is not None:
            decision_eval_n += 1
            if persona_obj.get("decision") == expected:
                decision_correct += 1

        acts = persona_obj.get("memory_actions") or []
        if not isinstance(acts, list):
            acts = []
        total_actions += len(acts)

        # apply memory actions to fresh memory (公平のため毎回初期メモリ)
        mem0 = {"stm": {"turns": []}, "work": {"goal": "", "constraints": [], "facts": []}, "ltm": {"items": []}}
        mem_after, action_results = apply_memory_actions(
            mem0,
            acts,
            last_user_text=text,
            last_persona_reply=(persona_obj.get("reply") or ""),
        )

        # pollution (簡易)
        mem_pollution += len(action_results.get("rejected") or []) + len(action_results.get("skipped_duplicates") or [])

        # retrieve metrics (もしmemory_actionsにretrieveが入ってた場合のみ)
        for item in action_results.get("retrieve") or []:
            retrieve_count += 1
            hits = (((item.get("result") or {}).get("hits")) or [])
            if len(hits) == 0:
                unnecessary_retrieve += 1

    elapsed = time.time() - started

    metrics = {
        "ts_ms": int(time.time() * 1000),
        "variant": "lora_yomi_v1_json",
        "base_model": BASE_MODEL,
        "lora_adapter_dir": str(LORA_ADAPTER_DIR),
        "eval_path": str(EVAL_PATH),
        "n_cases": total,
        "ok_rate": safe_div(ok, total),
        "invalid_json_rate": safe_div(invalid_json, total),
        "decision_eval_n": decision_eval_n,
        "decision_accuracy": safe_div(decision_correct, decision_eval_n),
        "forced_decision_rate": safe_div(forced_decision, total),
        "obedience_drop_rate": safe_div(dropped_actions, max(total_actions, 1)),
        "memory_pollution_rate": safe_div(mem_pollution, max(total_actions, 1)),
        "unnecessary_retrieve_rate": safe_div(unnecessary_retrieve, max(retrieve_count, 1)),
        "retrieve_count": retrieve_count,
        "elapsed_sec": elapsed,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"metrics_lora_{int(time.time())}.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote:", out_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
