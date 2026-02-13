import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "experiments" / "eval_100cases.router.jsonl"
OUT_METRICS = BASE_DIR / "runs" / "metrics_latest.json"
API_BASE = "http://127.0.0.1:8011"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def main() -> None:
    cases = load_jsonl(EVAL_PATH)

    total = 0
    ok = 0
    invalid_json = 0

    decision_eval_n = 0
    decision_correct = 0

    router_eval_n = 0
    router_correct = 0

    total_actions = 0
    dropped_actions = 0
    forced_decision = 0

    mem_pollution = 0

    unnecessary_retrieve = 0
    retrieve_count = 0

    retrieve_attempted = 0
    retrieve_skipped_by_gate = 0
    retrieve_executed = 0
    retrieve_hits_total = 0
    retrieve_nonempty_results = 0

    sample_errors: List[str] = []

    # --- persona別集計（routed_persona_idで集計） ---
    # 取りたい指標だけ入れる（増やしすぎない）
    per: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: {
        "n": 0,
        "decision_eval_n": 0,
        "decision_correct": 0,
        "router_eval_n": 0,
        "router_correct": 0,
        "retrieve_executed": 0,
        "retrieve_nonempty": 0,
        "unnecessary_retrieve": 0,
        "mem_pollution": 0,
        "dropped_actions": 0,
        "total_actions": 0,
    })

    started = time.time()
    with httpx.Client(timeout=180.0) as client:
        for c in cases:
            total += 1

            text = c.get("text", "")
            persona_id = c.get("persona_id", "yomi_proxy_v0")
            emotion = c.get("emotion") or {"joy": 1.0}

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

            routed_pid = str(data.get("routed_persona_id") or "unknown")
            p = per[routed_pid]
            p["n"] += 1

            # --- decision ---
            expected_decision = c.get("expected_decision")
            got_decision = None
            persona_obj = data.get("persona")
            if isinstance(persona_obj, dict):
                got_decision = persona_obj.get("decision")

            if expected_decision is not None:
                decision_eval_n += 1
                p["decision_eval_n"] += 1
                if got_decision == expected_decision:
                    decision_correct += 1
                    p["decision_correct"] += 1

            # --- router ---
            expected_pid = c.get("expected_persona_id")
            if expected_pid is not None:
                router_eval_n += 1
                p["router_eval_n"] += 1
                if routed_pid == expected_pid:
                    router_correct += 1
                    p["router_correct"] += 1

            # --- obedience ---
            ob = data.get("obedience_report") or {}
            if isinstance(ob, dict) and ob.get("forced_decision"):
                forced_decision += 1
            dropped = len((ob.get("dropped_actions") or []) if isinstance(ob, dict) else [])
            dropped_actions += dropped
            p["dropped_actions"] += dropped

            # --- memory_action_results ---
            mar = data.get("memory_action_results") or {}
            if not isinstance(mar, dict):
                mar = {}

            skipped_dups = len(mar.get("skipped_duplicates") or [])
            rej = len(mar.get("rejected") or [])
            pollution = skipped_dups + rej
            mem_pollution += pollution
            p["mem_pollution"] += pollution

            # --- gate metrics ---
            gate_items = mar.get("retrieve_gate") or []
            skipped_items = mar.get("retrieve_skipped") or []
            executed_items = mar.get("retrieve") or []

            attempted = len(gate_items)
            skipped = len(skipped_items)
            executed = len(executed_items)
            if attempted == 0 and (executed > 0 or skipped > 0):
                attempted = executed + skipped

            retrieve_attempted += attempted
            retrieve_skipped_by_gate += skipped
            retrieve_executed += executed

            p["retrieve_executed"] += executed

            # --- retrieve quality ---
            for item in executed_items:
                retrieve_count += 1
                hits = (((item.get("result") or {}).get("hits")) or [])
                if len(hits) == 0:
                    unnecessary_retrieve += 1
                    p["unnecessary_retrieve"] += 1
                else:
                    retrieve_nonempty_results += 1
                    retrieve_hits_total += len(hits)
                    p["retrieve_nonempty"] += 1

            # --- total_actions ---
            merged = data.get("merged_actions") or []
            n_actions = len(merged) if isinstance(merged, list) else 0
            total_actions += n_actions
            p["total_actions"] += n_actions

    elapsed = time.time() - started

    retrieve_hit_rate = safe_div(retrieve_nonempty_results, max(retrieve_executed, 1))
    retrieve_avg_hits_per_exec = safe_div(retrieve_hits_total, max(retrieve_executed, 1))

    # persona別サマリを整形
    per_out: Dict[str, Any] = {}
    for pid, x in per.items():
        per_out[pid] = {
            "n": int(x["n"]),
            "decision_acc": safe_div(x["decision_correct"], max(x["decision_eval_n"], 1)),
            "router_acc": safe_div(x["router_correct"], max(x["router_eval_n"], 1)),
            "retrieve_exec": int(x["retrieve_executed"]),
            "retrieve_hit_rate": safe_div(x["retrieve_nonempty"], max(x["retrieve_executed"], 1)),
            "unnecessary_retrieve_rate": safe_div(x["unnecessary_retrieve"], max(x["retrieve_executed"], 1)),
            "memory_pollution_rate": safe_div(x["mem_pollution"], max(x["total_actions"], 1)),
            "obedience_drop_rate": safe_div(x["dropped_actions"], max(x["total_actions"], 1)),
        }

    metrics = {
        "ts_ms": int(time.time() * 1000),
        "api_base": API_BASE,
        "eval_path": str(EVAL_PATH).replace("\\", "/"),
        "n_cases": total,
        "ok_rate": safe_div(ok, total),
        "invalid_json_rate": safe_div(invalid_json, total),

        "decision_eval_n": decision_eval_n,
        "decision_accuracy": safe_div(decision_correct, decision_eval_n),

        "router_eval_n": router_eval_n,
        "router_accuracy": safe_div(router_correct, router_eval_n),

        "forced_decision_rate": safe_div(forced_decision, total),
        "obedience_drop_rate": safe_div(dropped_actions, max(total_actions, 1)),
        "memory_pollution_rate": safe_div(mem_pollution, max(total_actions, 1)),

        "unnecessary_retrieve_rate": safe_div(unnecessary_retrieve, max(retrieve_count, 1)),
        "retrieve_count": retrieve_count,

        "retrieve_attempted": retrieve_attempted,
        "retrieve_skipped_by_gate": retrieve_skipped_by_gate,
        "retrieve_executed": retrieve_executed,
        "retrieve_hit_rate": retrieve_hit_rate,
        "retrieve_avg_hits_per_exec": retrieve_avg_hits_per_exec,

        "elapsed_sec": elapsed,
        "sample_errors": sample_errors,

        # ★ここが今日の目玉
        "by_persona": per_out,
    }

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote:", OUT_METRICS)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
