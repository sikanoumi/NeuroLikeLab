import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

# ★ Router評価したいなら expected_persona_id 入りの eval を指定する
EVAL_PATH = BASE_DIR / "experiments" / "eval_100cases.router.jsonl"

OUT_METRICS = BASE_DIR / "runs" / "metrics_latest.json"
API_BASE = "http://127.0.0.1:8011"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    JSONL loader (BOM-safe).
    """
    if not path.exists():
        raise FileNotFoundError(f"eval file not found: {path}")

    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def post_persona(client: httpx.Client, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        r = client.post(f"{API_BASE}/persona", json=payload)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _bucket() -> Dict[str, Any]:
    # persona別集計（READMEの表②に貼る用）
    return {
        "n": 0,
        "decision_eval_n": 0,
        "decision_correct": 0,

        "router_eval_n": 0,
        "router_correct": 0,

        # AgeMem Gate v0
        "retrieve_attempted": 0,
        "retrieve_skipped_by_gate": 0,
        "retrieve_executed": 0,
        "retrieve_nonempty_results": 0,  # executed retrieves with len(hits)>0
        "retrieve_hits_total": 0,        # total hits across executed retrieves

        # memory pollution（簡易：skipped_duplicates + rejected）
        "mem_pollution": 0,

        # obedience
        "forced_decision": 0,
        "dropped_actions": 0,
        "total_actions": 0,

        # retrieve（existing）
        "retrieve_count": 0,
        "unnecessary_retrieve": 0,
    }


def main() -> None:
    cases = load_jsonl(EVAL_PATH)

    total = 0
    ok = 0
    invalid_json = 0

    # decision eval（expected_decision があるケースだけ）
    decision_eval_n = 0
    decision_correct = 0

    # router eval（expected_persona_id があるケースだけ）
    router_eval_n = 0
    router_correct = 0

    # obedience
    total_actions = 0
    dropped_actions = 0
    forced_decision = 0

    # memory
    mem_pollution = 0  # skipped_duplicates + rejected の合計（簡易）

    # retrieve (existing metric)
    unnecessary_retrieve = 0
    retrieve_count = 0

    # AgeMem Gate v0 metrics
    retrieve_attempted = 0
    retrieve_skipped_by_gate = 0
    retrieve_executed = 0
    retrieve_hits_total = 0  # total hits across executed retrieves
    retrieve_nonempty_results = 0  # executed retrieves with len(hits)>0

    # persona breakdown
    by_persona: Dict[str, Dict[str, Any]] = defaultdict(_bucket)

    # errors sample
    sample_errors: List[str] = []

    started = time.time()
    with httpx.Client(timeout=180.0) as client:
        for c in cases:
            total += 1

            text = c.get("text", "")
            persona_id = c.get("persona_id", "yomi_proxy_v0")
            emotion = c.get("emotion") or {"joy": 1.0}

            # ★ Router評価のため use_router を明示的にON
            payload = {
                "text": text,
                "emotion": emotion,
                "persona_id": persona_id,
                "use_router": True,
                "task": c.get("task", "default"),
            }

            data, err = post_persona(client, payload)
            if err or not isinstance(data, dict):
                invalid_json += 1
                if len(sample_errors) < 5:
                    sample_errors.append(err or "unknown_error")
                continue

            ok += 1

            # persona key（ここが表②のキー）
            routed_pid = data.get("routed_persona_id") or "unknown"
            pb = by_persona[routed_pid]
            pb["n"] += 1

            # -------------------------
            # decision_accuracy
            # -------------------------
            expected = c.get("expected_decision")
            got = None
            persona_obj = data.get("persona")
            if isinstance(persona_obj, dict):
                got = persona_obj.get("decision")

            if expected is not None:
                decision_eval_n += 1
                pb["decision_eval_n"] += 1
                if got == expected:
                    decision_correct += 1
                    pb["decision_correct"] += 1

            # -------------------------
            # router_accuracy
            # -------------------------
            expected_pid = c.get("expected_persona_id")
            if expected_pid is not None:
                router_eval_n += 1
                pb["router_eval_n"] += 1
                if routed_pid == expected_pid:
                    router_correct += 1
                    pb["router_correct"] += 1

            # -------------------------
            # obedience
            # -------------------------
            ob = data.get("obedience_report") or {}
            if isinstance(ob, dict) and ob.get("forced_decision"):
                forced_decision += 1
                pb["forced_decision"] += 1

            dropped = len((ob.get("dropped_actions") or []) if isinstance(ob, dict) else [])
            dropped_actions += dropped
            pb["dropped_actions"] += dropped

            # -------------------------
            # memory_action_results
            # -------------------------
            mar = data.get("memory_action_results") or {}
            if not isinstance(mar, dict):
                mar = {}

            skipped_dups = len(mar.get("skipped_duplicates") or [])
            rej = len(mar.get("rejected") or [])
            mem_pollution += (skipped_dups + rej)
            pb["mem_pollution"] += (skipped_dups + rej)

            # -------------------------
            # AgeMem Gate v0 counters
            # -------------------------
            gate_items = mar.get("retrieve_gate") or []
            skipped_items = mar.get("retrieve_skipped") or []
            executed_items = mar.get("retrieve") or []

            attempted = len(gate_items)
            skipped = len(skipped_items)
            executed = len(executed_items)

            # Fallback: if gate logs are missing but retrieve executed, approximate attempted
            if attempted == 0 and (executed > 0 or skipped > 0):
                attempted = executed + skipped

            retrieve_attempted += attempted
            retrieve_skipped_by_gate += skipped
            retrieve_executed += executed

            pb["retrieve_attempted"] += attempted
            pb["retrieve_skipped_by_gate"] += skipped
            pb["retrieve_executed"] += executed

            # -------------------------
            # retrieve metrics
            # -------------------------
            # retrieveが空振りだった割合（hits=0）
            for item in executed_items:
                retrieve_count += 1
                pb["retrieve_count"] += 1

                hits = (((item.get("result") or {}).get("hits")) or [])
                if len(hits) == 0:
                    unnecessary_retrieve += 1
                    pb["unnecessary_retrieve"] += 1
                else:
                    retrieve_nonempty_results += 1
                    retrieve_hits_total += len(hits)

                    pb["retrieve_nonempty_results"] += 1
                    pb["retrieve_hits_total"] += len(hits)

            # total_actions（merge_actions数）
            merged_actions = len(data.get("merged_actions") or [])
            total_actions += merged_actions
            pb["total_actions"] += merged_actions

    elapsed = time.time() - started

    # Gate-aware derived metrics（global）
    retrieve_hit_rate = safe_div(retrieve_nonempty_results, max(retrieve_executed, 1))
    retrieve_avg_hits_per_exec = safe_div(retrieve_hits_total, max(retrieve_executed, 1))

    metrics = {
        "ts_ms": int(time.time() * 1000),
        "api_base": API_BASE,
        "eval_path": str(EVAL_PATH).replace("\\", "/"),
        "n_cases": total,
        "ok_rate": safe_div(ok, total),
        "invalid_json_rate": safe_div(invalid_json, total),

        # decision
        "decision_eval_n": decision_eval_n,
        "decision_accuracy": safe_div(decision_correct, decision_eval_n),

        # router
        "router_eval_n": router_eval_n,
        "router_accuracy": safe_div(router_correct, router_eval_n),

        # obedience
        "forced_decision_rate": safe_div(forced_decision, total),
        "obedience_drop_rate": safe_div(dropped_actions, max(total_actions, 1)),

        # memory
        "memory_pollution_rate": safe_div(mem_pollution, max(total_actions, 1)),

        # retrieve (existing)
        "unnecessary_retrieve_rate": safe_div(unnecessary_retrieve, max(retrieve_count, 1)),
        "retrieve_count": retrieve_count,

        # AgeMem Gate v0
        "retrieve_attempted": retrieve_attempted,
        "retrieve_skipped_by_gate": retrieve_skipped_by_gate,
        "retrieve_executed": retrieve_executed,
        "retrieve_hit_rate": retrieve_hit_rate,
        "retrieve_avg_hits_per_exec": retrieve_avg_hits_per_exec,

        # misc
        "elapsed_sec": elapsed,
        "sample_errors": sample_errors,
    }

    # -------------------------
    # Persona breakdown: rows for README table②
    # -------------------------
    persona_rows: List[Dict[str, Any]] = []
    for pid, b in sorted(by_persona.items(), key=lambda x: (-x[1]["n"], x[0])):
        # decision per persona（expected_decision がある分だけ）
        d_acc = safe_div(b["decision_correct"], b["decision_eval_n"])

        # router per persona（expected_persona_id がある分だけ）
        r_acc = safe_div(b["router_correct"], b["router_eval_n"])

        # retrieve per persona（Gate-aware）
        p_hit_rate = safe_div(b["retrieve_nonempty_results"], max(b["retrieve_executed"], 1))
        p_avg_hits = safe_div(b["retrieve_hits_total"], max(b["retrieve_executed"], 1))

        # memory pollution per persona（globalと同じ定義：actionsで割る）
        p_pollution_rate = safe_div(b["mem_pollution"], max(b["total_actions"], 1))

        persona_rows.append({
            "persona": pid,
            "n": b["n"],
            "decision_accuracy": round(d_acc, 4),
            "router_accuracy": round(r_acc, 4),
            "retrieve_attempted": int(b["retrieve_attempted"]),
            "skipped_by_gate": int(b["retrieve_skipped_by_gate"]),
            "executed": int(b["retrieve_executed"]),
            "hit_rate": (round(p_hit_rate, 4) if b["retrieve_executed"] else None),
            "avg_hits_per_exec": (round(p_avg_hits, 3) if b["retrieve_executed"] else None),
            "memory_pollution_rate": round(p_pollution_rate, 4),
        })

    metrics["persona_breakdown"] = persona_rows

    # -------------------------
    # write metrics
    # -------------------------
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # -------------------------
    # print (README paste)
    # -------------------------
    print("Wrote:", OUT_METRICS)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n## Persona breakdown (paste into README)\n")
    print("| persona | n | decision_acc | router_acc | retrieve_attempted | skipped_by_gate | executed | hit_rate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in persona_rows:
        hr = "-" if r["hit_rate"] is None else f'{r["hit_rate"]:.4f}'
        print(
            f'| {r["persona"]} | {r["n"]} | {r["decision_accuracy"]:.4f} | {r["router_accuracy"]:.4f} | '
            f'{r["retrieve_attempted"]} | {r["skipped_by_gate"]} | {r["executed"]} | {hr} |'
        )


if __name__ == "__main__":
    main()
