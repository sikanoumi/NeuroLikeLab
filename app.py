# app.py (NeuroLikeLab 直下) - Router + Multi-persona (v0.3-router1)
import os
import json
import time
import hashlib
import copy
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.mix import compute_latent_state
from core.state import update_state, get_initial_state
from core.io import append_turn_log
from core.memory import load_memory, save_memory, apply_memory_actions

print("[DEBUG] app.py file:", __file__)

app = FastAPI()

# --- CORS（Next UI から叩けるように） ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VERSION = "v0.3-router1"

# ★ 起動カレントに依存しないための基準ディレクトリ
BASE_DIR = Path(__file__).resolve().parent

# --- NeuroLike state ---
_current_state: Dict[str, float] = get_initial_state()

# --- Ollama settings ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# ★ runs/ も絶対パスで固定（CWD事故防止）
RUNS_DIR = (BASE_DIR / "runs").resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ★ デバッグ：読み込んでいる core.memory のファイルパスを出す（必要時のみ）
# PowerShell: $env:DEBUG_IMPORT_PATHS="1"
if os.getenv("DEBUG_IMPORT_PATHS", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
    import core.memory as _mem_mod  # noqa: F401
    print("[DEBUG] core.memory file:", _mem_mod.__file__)


# =========================
# Text normalize (JP spacing)
# =========================
def _clean_jp_spaces(s: str) -> str:
    """
    JP spacing cleaner (strong):
    - 全角空白→半角
    - NBSP / ゼロ幅 / BOM / thin spaces を除去
    - 日本語文字(ひら/カタ/漢字)の間に挟まった空白を除去（"疲 労","楽 しい","不安 感"）
    - 連続スペース/タブを1つに
    - 前後strip
    """
    if not s:
        return s

    # normalize weird/invisible spaces
    s = s.replace("\u3000", " ")   # fullwidth space
    s = s.replace("\u00A0", " ")   # NBSP
    s = s.replace("\uFEFF", "")    # BOM
    s = s.replace("\u200B", "")    # zero-width space
    s = s.replace("\u2009", " ")   # thin space
    s = s.replace("\u202F", " ")   # narrow no-break space

    # remove spaces between JP chars
    s = re.sub(r"(?<=[ぁ-んァ-ヶ一-龠])\s+(?=[ぁ-んァ-ヶ一-龠])", "", s)

    # collapse remaining spaces/tabs
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()




class AnalyzeReq(BaseModel):
    text: str
    emotion: Optional[Dict[str, float]] = None


class PersonaReq(BaseModel):
    text: str
    emotion: Optional[Dict[str, float]] = None
    persona_id: str = "yomi_proxy_v0"
    task: str = "default"
    use_router: bool = True


@app.get("/health")
def health():
    # healthは短いので dict のままでOK（でも統一したいならJSONResponseでも可）
    return {"ok": True, "version": VERSION}


@app.post("/analyze")
def analyze(req: AnalyzeReq):
    """状態更新 + logs/turns.jsonl へ保存"""
    global _current_state

    turn_id = f"t_{int(time.time() * 1000)}"
    emotion = req.emotion or {"joy": 1.0}
    clean_text = _clean_jp_spaces(req.text)

    latent_state = compute_latent_state(emotion)
    state_before = dict(_current_state)
    state_after = update_state(state_before, latent_state)
    _current_state = state_after

    response = {
        "turn_id": turn_id,
        "text": clean_text,
        "emotion": emotion,
        "latent_state": latent_state,
        "state_before": state_before,
        "state_after": state_after,
        "version": VERSION,
    }
    append_turn_log(response)

    # ★ PowerShell文字化け対策（charset明示）
    return JSONResponse(content=response, media_type="application/json; charset=utf-8")


# =========================
# Router（state×task -> persona_id）
# =========================
def route_persona_id(state_after: Dict[str, float], latent_state: Dict[str, float], task: str = "default") -> str:
    """
    Minimal router (improved):
      - High stress (max of latent/state_after) -> safety_v0
      - brainstorm/ideation -> creative_v0
      - else -> action_v0
    """
    cortisol = max(
        float((state_after or {}).get("cortisol_like", 0.0)),
        float((latent_state or {}).get("cortisol_like", 0.0)),
    )
    threat = max(
        float((state_after or {}).get("threat_bias", 0.0)),
        float((latent_state or {}).get("threat_bias", 0.0)),
    )

    t = (task or "default").lower().strip()
    if cortisol >= 0.7 or threat >= 0.6:
        return "safety_v0"
    if t in {"brainstorm", "ideation", "creative"}:
        return "creative_v0"
    return "action_v0"


# =========================
# Prompt I/O + SHA
# =========================
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _build_persona_prompt(
    turn_id: str,
    persona_id: str,
    text: str,
    emotion: Dict[str, float],
    latent_state: Dict[str, float],
    state_before: Dict[str, float],
    state_after: Dict[str, float],
    memory_ctx: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Ollamaへ渡すプロンプト（JSONのみ出力強制） + promptメタ（sha等）"""
    analysis = {
        "emotion": emotion,
        "latent_state": latent_state,
        "state_before": state_before,
        "state_after": state_after,
    }

    schema = {
        "turn_id": turn_id,
        "persona_id": persona_id,
        "model": OLLAMA_MODEL,
        "policy_version": "p0",
        "reply": "ユーザーに返す文章（日本語）",
        "reason_one_line": "理由(一行・日本語)",
        "decision": "defer|propose|ask_clarify",
        "confidence": 0.0,
        "decision_basis": [{"type": "observation|interpretation|action", "text": "..."}],
        "memory_actions": [
            {"action": "add|retrieve|summarize|forget|update|none", "target": "stm|work|ltm", "note": "..."}
        ],
    }

    system_path = (BASE_DIR / "prompts" / "system_p0.txt").resolve()
    persona_path = (BASE_DIR / "prompts" / "persona" / f"{persona_id}.txt").resolve()

    if not system_path.exists():
        raise FileNotFoundError(f"system prompt not found: {system_path}")
    if not persona_path.exists():
        raise FileNotFoundError(f"persona prompt not found: {persona_path}")

    system_txt = _read_text(system_path)
    persona_txt = _read_text(persona_path)

    system_sha1 = _sha1_text(system_txt)
    persona_sha1 = _sha1_text(persona_txt)

    prompt = f"""
{system_txt}

{persona_txt}

以下のJSONスキーマに従って出力してください（値は埋めてください）:
{json.dumps(schema, ensure_ascii=False, indent=2)}

[入力テキスト]
{text}

[分析データ(JSON)]
{json.dumps(analysis, ensure_ascii=False)}

[メモリ文脈(JSON)]
{json.dumps(memory_ctx, ensure_ascii=False)}
""".strip()

    meta = {
        "prompt_version": "p0",
        "persona_prompt_id": persona_id,
        "prompt_system_path": system_path.as_posix(),
        "prompt_persona_path": persona_path.as_posix(),
        "prompt_system_sha1": system_sha1,
        "prompt_persona_sha1": persona_sha1,
    }
    return prompt, meta


def _memory_context(mem: Dict[str, Any], recent_n: int = 6) -> Dict[str, Any]:
    work = mem.get("work", {}) or {}
    stm_turns: List[Dict[str, Any]] = (mem.get("stm", {}) or {}).get("turns", []) or []
    recent = stm_turns[-recent_n:] if isinstance(stm_turns, list) else []

    return {
        "work_memo": {
            "goal": work.get("goal", ""),
            "constraints": (work.get("constraints", []) or [])[:10],
            "facts": (work.get("facts", []) or [])[-10:],
        },
        "recent_stm": recent,
        "ltm_size": len((mem.get("ltm", {}) or {}).get("items", []) or []),
    }


def _stable_json_hash(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_json_object(s: str) -> Tuple[Optional[str], Optional[str]]:
    if not s:
        return None, "empty_response"
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_braces_found"
    return s[start : end + 1].strip(), None


def _append_run_line(run_line: Dict[str, Any]) -> None:
    with (RUNS_DIR / "run_ollama_001.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_line, ensure_ascii=False) + "\n")


# =========================
# Obedience Control (min)
# =========================
ALLOWED_DECISIONS = {"defer", "propose", "ask_clarify"}
ALLOWED_ACTIONS = {"add", "retrieve", "summarize", "update", "none"}
ALLOWED_TARGETS = {"stm", "work", "ltm"}


def _normalize_target_alias(t: str) -> str:
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

    p["reply"] = _clean_jp_spaces(str(p.get("reply") or "了解。まず状況を整えてから進めよう。"))
    p["reason_one_line"] = _clean_jp_spaces(str(p.get("reason_one_line") or "整合性を優先。"))

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
        tgt = _normalize_target_alias(str(tgt_raw))
        note = _clean_jp_spaces((a.get("note") or "").strip())

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


# =========================
# Phase6(min): policy（latent発火 + retrieve短縮）
# =========================
def memory_action_policy(
    emotion: Dict[str, float],
    state_after: Dict[str, float],
    latent_state: Dict[str, float],
    user_text: str,
    stm_turns_count: int,
) -> List[Dict[str, Any]]:
    anxiety = float((emotion or {}).get("anxiety", 0.0))
    confidence = float((emotion or {}).get("confidence", 0.0))
    fatigue = float((emotion or {}).get("fatigue", 0.0))

    cortisol_now = float((latent_state or {}).get("cortisol_like", 0.0))
    threat_now = float((latent_state or {}).get("threat_bias", 0.0))
    cortisol_acc = float((state_after or {}).get("cortisol_like", 0.0))
    threat_acc = float((state_after or {}).get("threat_bias", 0.0))

    high_stress = (
        (cortisol_now >= 0.8)
        or (threat_now >= 0.6)
        or (anxiety >= 0.7)
        or (cortisol_acc >= 0.7)
        or (threat_acc >= 0.6)
    )
    high_fatigue = fatigue >= 0.8
    high_conf = confidence >= 0.7

    actions: List[Dict[str, Any]] = []

    if high_stress:
        q = "休憩" if fatigue >= 0.8 else "ストレス"
        actions.append({"action": "retrieve", "target": "stm", "note": q})

        if int(stm_turns_count or 0) >= 8:
            actions.append({"action": "summarize", "target": "work", "note": ""})

    if high_conf:
        actions.append({"action": "retrieve", "target": "work", "note": "次の一手"})

    if high_fatigue:
        actions.append({"action": "add", "target": "stm", "note": "疲労が高いので休憩優先（policy）"})

    return actions


def merge_actions(policy_actions: List[Dict[str, Any]], llm_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for a in (policy_actions or []) + (llm_actions or []):
        act = (a.get("action") or "none").lower()
        tgt = (a.get("target") or "stm").lower()
        note = _clean_jp_spaces((a.get("note") or "").strip())
        key = (act, tgt, note)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"action": act, "target": tgt, "note": note})

    return merged


@app.post("/persona")
def persona(req: PersonaReq) -> JSONResponse:
    global _current_state

    debug_ja: List[str] = ["Phase5/6: /persona 開始"]

    turn_id = f"t_{int(time.time() * 1000)}"
    emotion = req.emotion or {"joy": 1.0}
    clean_text = _clean_jp_spaces(req.text)

    # ---- NeuroLike update ----
    latent_state = compute_latent_state(emotion)
    state_before = dict(_current_state)
    state_after = update_state(state_before, latent_state)
    _current_state = state_after

    analyze_payload = {
        "turn_id": turn_id,
        "text": clean_text,  # ★必ず正規化済み
        "emotion": emotion,
        "latent_state": latent_state,
        "state_before": state_before,
        "state_after": state_after,
        "version": VERSION,
    }
    append_turn_log(analyze_payload)
    debug_ja.append("NeuroLike: state 更新 & logs 追記 OK")

    # ---- Router ----
    routed_persona_id = req.persona_id
    if bool(req.use_router):
        routed_persona_id = route_persona_id(state_after, latent_state, task=req.task)
    debug_ja.append(f"Router: use_router={req.use_router} task={req.task} persona={routed_persona_id}")

    # ---- Memory (before) ----
    mem_before = load_memory()
    mem_before_snapshot = copy.deepcopy(mem_before)
    stm_turns = ((mem_before.get("stm") or {}).get("turns") or [])
    stm_turns_count = len(stm_turns) if isinstance(stm_turns, list) else 0
    mem_ctx = _memory_context(mem_before, recent_n=6)
    debug_ja.append(f"Memory: 読み込み OK (stm_turns={stm_turns_count})")

    # ---- policy actions ----
    policy_actions = memory_action_policy(
        emotion=emotion,
        state_after=state_after,
        latent_state=latent_state,
        user_text=clean_text,
        stm_turns_count=stm_turns_count,
    )
    debug_ja.append(f"Policy: actions 自動生成 (n={len(policy_actions)})")

    # ---- Call Ollama ----
    prompt, prompt_meta = _build_persona_prompt(
        turn_id=turn_id,
        persona_id=routed_persona_id,
        text=clean_text,  # ★必ず正規化済み
        emotion=emotion,
        latent_state=latent_state,
        state_before=state_before,
        state_after=state_after,
        memory_ctx=mem_ctx,
    )
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    persona_raw = ""
    ollama_error: Optional[str] = None

    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post(f"{OLLAMA_URL}/api/generate", json=body)
            r.raise_for_status()
            data = r.json()
        persona_raw = (data.get("response") or "").strip()
        debug_ja.append("Ollama: 応答取得 OK")
    except Exception as e:
        ollama_error = str(e)
        debug_ja.append(f"Ollama: 失敗 ({ollama_error})")

    # ---- Parse JSON ----
    persona_obj: Optional[dict] = None
    parse_error: Optional[str] = None
    extracted_json: Optional[str] = None
    extracted_json_clean: Optional[str] = None

    if ollama_error is None:
        extracted_json, extract_err = _extract_json_object(persona_raw)
        if extract_err:
            parse_error = extract_err
            debug_ja.append(f"Parse: 失敗（{parse_error}）")
        else:
            try:
                persona_obj = json.loads(extracted_json)
                debug_ja.append("Parse: JSONパース OK")
            except Exception as e:
                parse_error = str(e)
                debug_ja.append(f"Parse: JSONパース失敗（{parse_error}）")

    persona_parsed_ok = persona_obj is not None

    # ---- Obedience sanitize ----
    obedience_report: Optional[dict] = None
    if persona_parsed_ok:
        persona_obj, obedience_report = validate_and_sanitize_persona(persona_obj)
        debug_ja.append(
            f"Obedience: kept={obedience_report['kept_actions']} dropped={len(obedience_report['dropped_actions'])}"
        )
        # sanitize後のJSONをログ用に作り直す（“綺麗な方”を残す）
        extracted_json_clean = json.dumps(persona_obj, ensure_ascii=False, indent=2)

    # ---- Merge actions ----
    llm_actions: List[Dict[str, Any]] = []
    if persona_parsed_ok:
        llm_actions = (persona_obj.get("memory_actions") or [])
        debug_ja.append(f"LLM: memory_actions 取得 (n={len(llm_actions)})")

    merged_actions = merge_actions(policy_actions, llm_actions)
    debug_ja.append(f"Merge: policy+llm actions = {len(merged_actions)}")

    # ---- Apply actions ----
    mem_after = mem_before
    action_results: Dict[str, Any] = {"skipped": True, "reason": "persona_json_parse_failed"}

    if persona_parsed_ok:
        try:
            mem_after, action_results = apply_memory_actions(
                mem_before_snapshot,
                merged_actions,
                last_user_text=clean_text,  # ★必ず正規化済み
                last_persona_reply=(persona_obj.get("reply", "") or ""),
                persona_id=routed_persona_id,
                task=req.task,
            )
            save_memory(mem_after)
            debug_ja.append("Memory: actions 適用 & 保存 OK")
        except Exception as e:
            action_results = {"skipped": True, "reason": "apply_memory_actions_failed", "error": str(e)}
            mem_after = mem_before
            debug_ja.append(f"Memory: actions 適用失敗（{str(e)}）")
    else:
        merged_actions = []
        debug_ja.append("Parse失敗のため memory_actions は適用しない（安全側）")

    # ---- Hash ----
    mem_after_disk = load_memory()
    memory_before_hash = _stable_json_hash(mem_before)
    memory_after_hash = _stable_json_hash(mem_after_disk)

    if memory_before_hash != memory_after_hash:
        debug_ja.append("Hash: before/after 差分あり（更新を検知）")
    else:
        debug_ja.append("Hash: before/after が同一（更新なし）")

    # ---- runs ----
    run_line = {
        "turn_id": turn_id,
        "task": req.task,
        "use_router": req.use_router,
        "persona_id": req.persona_id,
        "routed_persona_id": routed_persona_id,
        "ollama_model": OLLAMA_MODEL,
        "prompt_meta": prompt_meta,
        "analyze": analyze_payload,
        "ollama_error": ollama_error,
        "persona_raw": persona_raw,
        # “綺麗なJSON” をログに残す
        "persona_extracted_json": extracted_json,
        "persona_extracted_json_clean": extracted_json_clean,
        "persona_parsed_ok": persona_parsed_ok,
        "persona_parse_error": parse_error,
        "obedience_report": obedience_report,
        "policy_actions": policy_actions,
        "llm_actions": llm_actions,
        "merged_actions": merged_actions,
        "memory_before_hash": memory_before_hash,
        "memory_after_hash": memory_after_hash,
        "memory_before": mem_before,
        "memory_after": mem_after,
        "memory_action_results": action_results,
        "debug_ja": debug_ja,
        "ts_ms": int(time.time() * 1000),
        "version": VERSION,
    }

    try:
        _append_run_line(run_line)
    except Exception as e:
        run_line["runs_write_error"] = str(e)
        debug_ja.append(f"runs: 書き込み失敗（{str(e)}）")

    # ---- response ----
    resp = {
        "turn_id": turn_id,
        "analyze": analyze_payload,
        "task": req.task,
        "use_router": req.use_router,
        "persona_id": req.persona_id,
        "routed_persona_id": routed_persona_id,
        "persona": persona_obj,
        "persona_raw": persona_raw,
        "persona_extracted_json": extracted_json,
        "persona_extracted_json_clean": extracted_json_clean,
        "ollama_error": ollama_error,
        "persona_parse_error": parse_error,
        "obedience_report": obedience_report,
        "policy_actions": policy_actions,
        "merged_actions": merged_actions,
        "memory_action_results": action_results,
        "memory_before_hash": memory_before_hash,
        "memory_after_hash": memory_after_hash,
        "prompt_meta": prompt_meta,
        "debug_ja": debug_ja,
        "version": VERSION,
    }

    # ★ PowerShell文字化け対策（charset明示）
    return JSONResponse(content=resp, media_type="application/json; charset=utf-8")
