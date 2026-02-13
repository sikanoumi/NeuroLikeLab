import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "experiments" / "eval_100cases.jsonl"
DST = BASE_DIR / "experiments" / "eval_100cases.clean.jsonl"
BAD = BASE_DIR / "runs" / "eval_bad_lines.json"

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"SRC not found: {SRC}")

    # 既存の clean を消して作り直し（追記事故防止）
    if DST.exists():
        DST.unlink()

    bad = []
    ok = 0
    total = 0

    lines = SRC.read_text(encoding="utf-8-sig").splitlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)

            # expect.decision -> expected_decision に寄せる
            if "expected_decision" not in obj:
                exp = obj.get("expect")
                if isinstance(exp, dict) and isinstance(exp.get("decision"), str):
                    obj["expected_decision"] = exp["decision"]

            # 必須の最低限
            if not isinstance(obj.get("text"), str):
                raise ValueError("missing text")
            if not isinstance(obj.get("persona_id"), str):
                obj["persona_id"] = "yomi_proxy_v0"
            if not isinstance(obj.get("emotion"), dict):
                obj["emotion"] = {"joy": 1.0}

            with DST.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            ok += 1

        except Exception as e:
            bad.append({"line_no": idx, "error": str(e), "line": line[:250]})

    BAD.parent.mkdir(parents=True, exist_ok=True)
    BAD.write_text(json.dumps(bad, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("SRC:", SRC)
    print("DST:", DST)
    print("BAD:", BAD)
    print(f"total_nonempty={total} ok={ok} bad={len(bad)}")

if __name__ == "__main__":
    main()
