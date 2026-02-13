# core/memory.py
# - Safety hardened + JP spacing normalize + CWD事故防止（絶対パス化）
# - 変な空白（例: "疲 労", "休 憩", "ホルモ ン"）をメモリ/ログに残さない
# - can_retrieve の q_len 判定も「正規化後」の長さで評価
# - summarize の work 汚染ゲート維持
# - ★summarize時点でSTM数を再取得（add等で増えた分を反映）

import json
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =========================
# Path safety (CWD事故防止)
# =========================
# core/memory.py の1つ上がプロジェクトルート（NeuroLikeLab）想定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEM_DIR = (PROJECT_ROOT / "memory").resolve()
STM_PATH = MEM_DIR / "stm.json"
WORK_PATH = MEM_DIR / "work_memo.json"
LTM_PATH = MEM_DIR / "ltm.json"


# =========================
# Text normalize (JP spacing)
# =========================
def _clean_jp_spaces(s: str) -> str:
    """
    - 全角空白→半角
    - 日本語文字(ひら/カタ/漢字)の間に挟まった半角スペースを除去
      例: "疲 労", "取っ て", "更 新"
    - 連続スペースを1つに
    """
    if not s:
        return s
    s = s.replace("\u3000", " ")
    s = re.sub(r"(?<=[ぁ-んァ-ヶ一-龠])\s+(?=[ぁ-んァ-ヶ一-龠])", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _safe_str(x: Any) -> str:
    # JSONからの読み込みで想定外が来ても落ちない
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _trim(s: str, max_len: int = 500) -> str:
    s = _safe_str(s)
    if len(s) > max_len:
        return s[:max_len]
    return s


# =========================
# JSON I/O
# =========================
def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        # WindowsでのBOM混入に強い
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        # 壊れたJSONでも落とさず復旧（安全側）
        return default


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_memory() -> Dict[str, Any]:
    """Load STM/WORK/LTM."""
    stm = _load_json(STM_PATH, {"turns": []})
    work = _load_json(WORK_PATH, {"goal": "", "constraints": [], "facts": []})
    ltm = _load_json(LTM_PATH, {"items": []})

    # 追加の安全整形（想定外型に備える）
    if not isinstance(stm, dict):
        stm = {"turns": []}
    if not isinstance(work, dict):
        work = {"goal": "", "constraints": [], "facts": []}
    if not isinstance(ltm, dict):
        ltm = {"items": []}

    return {"stm": stm, "work": work, "ltm": ltm}


def save_memory(mem: Dict[str, Any]) -> None:
    """Save STM/WORK/LTM."""
    _save_json(STM_PATH, mem.get("stm", {"turns": []}))
    _save_json(WORK_PATH, mem.get("work", {"goal": "", "constraints": [], "facts": []}))
    _save_json(LTM_PATH, mem.get("ltm", {"items": []}))


# =========================
# Structure helpers
# =========================
def _ensure_structure(mem: Dict[str, Any]) -> None:
    mem.setdefault("stm", {"turns": []})
    mem.setdefault("work", {"goal": "", "constraints": [], "facts": []})
    mem.setdefault("ltm", {"items": []})

    if not isinstance(mem.get("stm"), dict):
        mem["stm"] = {"turns": []}
    if not isinstance(mem.get("work"), dict):
        mem["work"] = {"goal": "", "constraints": [], "facts": []}
    if not isinstance(mem.get("ltm"), dict):
        mem["ltm"] = {"items": []}

    mem["stm"].setdefault("turns", [])
    mem["work"].setdefault("goal", "")
    mem["work"].setdefault("constraints", [])
    mem["work"].setdefault("facts", [])
    mem["ltm"].setdefault("items", [])

    if not isinstance(mem["stm"].get("turns"), list):
        mem["stm"]["turns"] = []
    if not isinstance(mem["work"].get("constraints"), list):
        mem["work"]["constraints"] = []
    if not isinstance(mem["work"].get("facts"), list):
        mem["work"]["facts"] = []
    if not isinstance(mem["ltm"].get("items"), list):
        mem["ltm"]["items"] = []


def _truncate_stm(stm: Dict[str, Any], keep_last: int = 12) -> None:
    turns = stm.get("turns", [])
    if isinstance(turns, list) and len(turns) > keep_last:
        stm["turns"] = turns[-keep_last:]


def _normalize_target(target: str) -> str:
    """
    LLMが返す target の表記ゆれを吸収する（Phase6で重要）
    期待値: stm / work / ltm
    """
    t = (target or "stm").lower().strip()
    alias = {
        # work系
        "work_memo": "work",
        "workmemo": "work",
        "work-memo": "work",
        "memo": "work",
        "worknote": "work",
        # stm系
        "short_term": "stm",
        "short": "stm",
        "stm_turns": "stm",
        "recent_stm": "stm",
        # ltm系
        "long_term": "ltm",
        "long": "ltm",
        "ltm_items": "ltm",
    }
    return alias.get(t, t)


# =========================
# Retrieve Gate (AgeMem)
# =========================
def can_retrieve(
    *,
    persona_id: str = "",
    task: str = "default",
    query: str = "",
    stm_turns_count: int = 0,
    cooldown_turns: int = 2,
    recently_retrieved: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    AgeMem Gate v0: decide whether to execute retrieve.

    Returns: (allowed, meta)
      meta: {allowed, reason, rule, task, persona_id, q_len, stm_turns_count}
    """
    pid = (persona_id or "").strip()
    t = (task or "default").lower().strip()

    q_norm = _clean_jp_spaces(_safe_str(query))
    q_len = len(q_norm)
    # ---- Generic short query skip (v0.1) ----
    # policyがよく出す短い固定語はヒットしにくく、空振りを増やすので原則skip
    GENERIC_SHORT = {"休憩", "ストレス", "次の一手"}
    if q_norm in GENERIC_SHORT:
        return False, {
            "allowed": False,
            "reason": "generic_query_skip",
            "rule": "HARD:generic_short_query",
            "task": t,
            "persona_id": pid,
            "q_len": q_len,
            "stm_turns_count": stm_turns_count,
        }

    # ---- Hard skips (v0) ----
    if q_len < 30:
        return False, {
            "allowed": False,
            "reason": "query_too_short",
            "rule": "HARD:q_len<30",
            "task": t,
            "persona_id": pid,
            "q_len": q_len,
            "stm_turns_count": stm_turns_count,
        }

    if t in {"brainstorm", "ideation", "creative"}:
        return False, {
            "allowed": False,
            "reason": "creative_task_skip",
            "rule": "HARD:task_in_creative",
            "task": t,
            "persona_id": pid,
            "q_len": q_len,
            "stm_turns_count": stm_turns_count,
        }

    if recently_retrieved:
        return False, {
            "allowed": False,
            "reason": "cooldown_recent_retrieve",
            "rule": f"HARD:cooldown_turns={cooldown_turns}",
            "task": t,
            "persona_id": pid,
            "q_len": q_len,
            "stm_turns_count": stm_turns_count,
        }

    return True, {
        "allowed": True,
        "reason": "allowed_by_default",
        "rule": "ALLOW:default",
        "task": t,
        "persona_id": pid,
        "q_len": q_len,
        "stm_turns_count": stm_turns_count,
    }


def retrieve(mem: Dict[str, Any], query: str, limit: int = 5) -> Dict[str, Any]:
    """Very simple retrieval: substring match over work/facts + ltm items + recent stm."""
    q = _clean_jp_spaces(_safe_str(query))
    if not q:
        return {"hits": []}

    hits: List[Dict[str, Any]] = []

    # Work memo
    work = mem.get("work", {}) or {}
    for f in (work.get("facts", []) or []):
        ftxt = _clean_jp_spaces(_safe_str(f))
        if q in ftxt:
            hits.append({"source": "work.facts", "text": ftxt})
    for c in (work.get("constraints", []) or []):
        ctxt = _clean_jp_spaces(_safe_str(c))
        if q in ctxt:
            hits.append({"source": "work.constraints", "text": ctxt})

    # LTM
    ltm = mem.get("ltm", {}) or {}
    for it in (ltm.get("items", []) or []):
        text = _clean_jp_spaces(_safe_str((it or {}).get("text", "")))
        if q in text:
            hits.append({"source": "ltm", "text": text})

    # STM (recent)
    stm = mem.get("stm", {}) or {}
    for t in reversed(stm.get("turns", []) or []):
        text = _clean_jp_spaces(_safe_str((t or {}).get("text", "")))
        if q in text:
            hits.append({"source": "stm", "text": text})

    return {"hits": hits[: max(1, int(limit or 5))]}


def summarize_stm_into_work(mem: Dict[str, Any], keep_last: int = 8) -> str:
    """
    Minimal summarization:
    - Take older STM turns (except last keep_last) and append a coarse summary string into work.facts.
    - Then truncate STM.
    """
    stm = mem.get("stm", {}) or {}
    turns = stm.get("turns", []) or []
    if not isinstance(turns, list) or len(turns) <= keep_last:
        return ""

    older = turns[:-keep_last]
    parts: List[str] = []
    for t in older:
        txt = _clean_jp_spaces(_safe_str((t or {}).get("text", "")))
        if txt:
            parts.append(_trim(txt, 60))

    summary = _trim(" / ".join(parts), 500)
    summary = _clean_jp_spaces(summary)

    if summary:
        work = mem.get("work", {}) or {}
        work.setdefault("facts", [])
        work["facts"].append(f"[stm_summary] {summary}")
        mem["work"] = work

    stm["turns"] = turns[-keep_last:]
    mem["stm"] = stm
    return summary


def _add_to_target(mem: Dict[str, Any], target: str, note: str, last_user_text: str = "") -> str:
    """
    Shared helper for add/update (Phase5).
    Returns the stored text.

    Special return:
      - "__DUPLICATE__" : 連続重複のため追加しない（stm/work で使用）
    """
    target = _normalize_target(target)

    # ★ここで正規化してから保存（汚い空白を永続化しない）
    note_clean = _clean_jp_spaces(_safe_str(note))
    last_user_clean = _clean_jp_spaces(_safe_str(last_user_text))

    text = note_clean or last_user_clean
    text = _trim(text, 500)
    text = _clean_jp_spaces(text)

    if not text:
        return ""

    if target == "stm":
        mem["stm"].setdefault("turns", [])
        turns = mem["stm"]["turns"]

        last_text = _clean_jp_spaces(_safe_str(turns[-1].get("text", ""))) if turns else ""
        new_text = text

        if last_text == new_text:
            return "__DUPLICATE__"

        turns.append({"text": text})
        _truncate_stm(mem["stm"], keep_last=12)

    elif target == "work":
        mem["work"].setdefault("facts", [])
        facts = mem["work"]["facts"]

        last_fact = _clean_jp_spaces(_safe_str(facts[-1])) if facts else ""
        new_fact = text
        if last_fact == new_fact:
            return "__DUPLICATE__"

        facts.append(text)

    elif target == "ltm":
        mem["ltm"].setdefault("items", [])
        mem["ltm"]["items"].append({"text": text})

    else:
        return ""

    return text


def apply_memory_actions(
    mem: Dict[str, Any],
    actions: List[Dict[str, Any]],
    last_user_text: str = "",
    last_persona_reply: str = "",
    persona_id: str = "",
    task: str = "default",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute memory actions (Phase5/6 minimal).

    Supported:
      - add (stm/work/ltm) : append
      - update (stm/work/ltm) : Phase5暫定で add と同等扱い（真の更新はPhase6以降）
      - retrieve : returns hits, does not modify memory
      - summarize : summarize stm into work
      - none : no-op

    Returns: (updated_mem, action_results)
    """
    mem2: Dict[str, Any] = copy.deepcopy(mem)
    _ensure_structure(mem2)

    SUMMARIZE_MIN_TURNS = 8

    action_results: Dict[str, Any] = {
        "applied": [],
        "rejected": [],
        "retrieve": [],
        "retrieve_gate": [],
        "retrieve_skipped": [],
        "summaries": [],
        "update_as_add": [],
        "skipped_duplicates": [],
        "unhandled": [],
    }

    # ★入力も正規化（結果ログの note にも反映される）
    last_user_text_clean = _clean_jp_spaces(_safe_str(last_user_text))
    last_persona_reply_clean = _clean_jp_spaces(_safe_str(last_persona_reply))

    for a in actions or []:
        try:
            act = (a.get("action") or "none").lower().strip()
            target_raw = a.get("target") or "stm"
            target = _normalize_target(_safe_str(target_raw))
            note_raw = a.get("note") or ""
            note = _clean_jp_spaces(_safe_str(note_raw))
            note = _trim(note, 500)

            if act == "none":
                continue

            if act == "add":
                stored = _add_to_target(mem2, target, note, last_user_text=last_user_text_clean)
                if stored == "__DUPLICATE__":
                    action_results["skipped_duplicates"].append({"action": "add", "target": target, "note": note})
                elif stored:
                    action_results["applied"].append(
                        {"action": "add", "target": target, "stored_text": stored, "note": note}
                    )
                else:
                    action_results["rejected"].append(
                        {"action": "add", "target": target_raw, "note": note, "reason": "empty_text_or_unknown_target"}
                    )

            elif act == "update":
                stored = _add_to_target(mem2, target, note, last_user_text=last_user_text_clean)
                if stored == "__DUPLICATE__":
                    action_results["skipped_duplicates"].append({"action": "update", "target": target, "note": note})
                elif stored:
                    action_results["update_as_add"].append({"target": target, "stored_text": stored, "note": note})
                    action_results["applied"].append(
                        {"action": "update_as_add", "target": target, "stored_text": stored, "note": note}
                    )
                else:
                    action_results["rejected"].append(
                        {"action": "update", "target": target_raw, "note": note, "reason": "empty_text_or_unknown_target"}
                    )

            elif act == "retrieve":
                q = note or last_user_text_clean or last_persona_reply_clean
                q = _clean_jp_spaces(q)
                q = _trim(q, 500)

                # Cooldown approximation:
                recent_turns = (mem2.get("stm", {}) or {}).get("turns", []) or []
                recent_texts = [
                    _safe_str(x.get("text", ""))
                    for x in (recent_turns[-2:] if isinstance(recent_turns, list) else [])
                ]
                recently_retrieved = any("[retrieve]" in s for s in recent_texts)

                # ★ここも「その時点」のstm_turns_countを反映（retrieve gate のメタ整合）
                cur_turns = (mem2.get("stm", {}) or {}).get("turns", []) or []
                cur_count = len(cur_turns) if isinstance(cur_turns, list) else 0

                allowed, meta = can_retrieve(
                    persona_id=persona_id,
                    task=task,
                    query=q,
                    stm_turns_count=cur_count,
                    cooldown_turns=2,
                    recently_retrieved=recently_retrieved,
                )
                action_results["retrieve_gate"].append({"query": q, "meta": meta})

                if not allowed:
                    action_results["retrieve_skipped"].append({"query": q, "meta": meta})
                    continue

                result = retrieve(mem2, q, limit=5)
                action_results["retrieve"].append({"query": q, "result": result, "gate": meta})

                # Cooldown marker
                try:
                    mem2["stm"].setdefault("turns", [])
                    mem2["stm"]["turns"].append({"text": "[retrieve] executed"})
                    _truncate_stm(mem2["stm"], keep_last=12)
                except Exception:
                    pass

            elif act == "summarize":
                # ★その時点のSTM量でゲート判定（add等で増えた分を反映）
                cur_turns = (mem2.get("stm", {}) or {}).get("turns", []) or []
                cur_count = len(cur_turns) if isinstance(cur_turns, list) else 0

                if cur_count < SUMMARIZE_MIN_TURNS:
                    action_results["rejected"].append(
                        {
                            "action": "summarize",
                            "target": "work",
                            "note": note,
                            "reason": f"summarize_gate: stm_turns_count({cur_count}) < {SUMMARIZE_MIN_TURNS}",
                        }
                    )
                    continue

                summary = summarize_stm_into_work(mem2, keep_last=8)
                summary = _clean_jp_spaces(summary)
                if summary:
                    action_results["summaries"].append(summary)
                    action_results["applied"].append(
                        {"action": "summarize", "target": "work", "stored_text": summary, "note": note}
                    )
                else:
                    action_results["rejected"].append(
                        {"action": "summarize", "target": "work", "note": note, "reason": "nothing_to_summarize"}
                    )

            else:
                action_results["unhandled"].append({"action": act, "target": target_raw, "note": note})

        except Exception as e:
            action_results["rejected"].append(
                {"action": a.get("action"), "target": a.get("target"), "note": a.get("note"), "reason": str(e)}
            )

    return mem2, action_results
