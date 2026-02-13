# NeuroLikeLab (v0.3-router1)
**text → emotion → latent_state(6-axis) → state_update → persona decision → JSONL logs → eval(metrics)**  
NeuroLikeLab is a minimal, reproducible experimental harness for **“persona as policy”** research.

## 完成定義B（研究として完結）
**3人格（Safety/Action/Creative）**を用意し、**Router（state×task）**が人格を選択。  
**MemGPT（stm/work/ltm）＋AgeMem（retrieve gate）**を統合し、人格ごとに memory / retrieval policy が変わることを **評価(metrics) と証拠(runs)** で示す。

## What this project demonstrates
- **Observable pipeline**: all steps are logged in JSONL (UTF-8)
- **Evaluation loop**: fixed eval cases → metrics JSON → runs evidence
- **Persona comparison**: Prompt persona (Ollama) vs **LoRA-fixed persona** (WSL + LLaMA-Factory)
- **Multi-persona Router**: state×task → persona selection (safety/action/creative) + evaluation

---

## Evidence (fixed snapshot)
- `runs/metrics_router100_20260212.json` （本READMEの数値の根拠）
- Standard bench: `experiments/eval_100cases.router.jsonl`

---

## ① Variant差分（全体）
> Portfolio-facing evidence. All variants are evaluated on the same eval set.

| Condition | n_cases | ok_rate | invalid_json | decision_acc | forced_decision | obedience_drop | memory_pollution | unnecessary_retrieve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before (Ollama + policy/obedience) | 100 | 1.00 | 0.00 | 0.56 | 0.00 | 0.0153 | 0.0728 | 0.5556 |
| After (Policy tuning: gate2) | 100 | 1.00 | 0.00 | 0.58 | 0.00 | 0.0074 | 0.1037 | 0.5132 |
| After (LoRA persona v1: yomi_lora_v1_json) | 100 | 1.00 | 0.00 | 0.55 | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| After (LoRA persona v2: yomi_lora_v2_json, label-aligned) | 100 | 1.00 | 0.00 | 1.00 | 0.00 | 0.0000 | 0.0000 | 1.0000 |

**Notes**
- `decision_acc` uses `expected_decision` in eval cases.
- `unnecessary_retrieve` is computed from retrieval calls where `hits=0`.
- LoRA eval currently measures **LoRA output consistency** (JSON validity + decision) without mixing server-side policy actions.
- In LoRA eval, memory is initialized as empty (`mem0`) for fairness; retrieve actions may yield `hits=0` and inflate `unnecessary_retrieve_rate`.

---

## ② Persona別差分（Router100 / 2026-02-12）
> This table is the key proof that **retrieval policy differs by persona** (via routing + gate logs).

| persona | n | decision_acc | router_acc | retrieve_attempted | skipped_by_gate | executed | hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| action_v0 | 65 | 0.8000 | 1.0000 | 41 | 41 | 0 | - |
| safety_v0 | 35 | 0.7429 | 1.0000 | 36 | 36 | 0 | - |

---

## 解釈（3行）
- `decision_acc=0.78` の主因は **defer / ask_clarify 境界**の揺れ（曖昧入力で分岐が割れる）。
- `retrieve_executed=0` は **AgeMem gate が全件で発火**して retrieve を抑制していることを示す（不要なretrieve回避側に倒れている）。
- 次の改善は **gate閾値（q_len等）/ task条件**の調整、または **query正規化**で executed を意図的に出して比較可能にする。

---

## Bench（標準ベンチ）
標準ベンチは `experiments/eval_100cases.router.jsonl`。  
`expected_decision / expected_persona_id / task` を含み、decision/router/memory/retrieve を同一ケースで評価する。

---

## Quickstart (Windows / PowerShell)

### Setup
```powershell
# in project root
python -m pip install -r requirements.txt


Setup

# in project root
python -m pip install -r requirements.txt

Start server (Ollama + FastAPI)

$env:OLLAMA_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="qwen3:8b"
python -m uvicorn app:app --host 127.0.0.1 --port 8011 --log-level info

Health check

irm http://127.0.0.1:8011/health

Call /persona (PowerShell UTF-8 safe)

chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$bodyObj = [ordered]@{
  text = "上司に詰められてる。今日中に方針を出せと言われた。正直いま判断が重い。"
  emotion = [ordered]@{ anxiety = 0.6; confidence = 0.3; fatigue = 0.7 }
  persona_id = "yomi_proxy_v0"
  use_router = $true
  task = "default"
}
$bodyJson  = $bodyObj | ConvertTo-Json -Depth 10
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)

irm http://127.0.0.1:8011/persona `
  -Method Post `
  -Body $bodyBytes `
  -ContentType "application/json; charset=utf-8"

Logs

Get-Content -Encoding utf8 .\runs\run_ollama_001.jsonl -Tail 1
Get-Content -Encoding utf8 .\runs\metrics_latest.json -Tail 80

Project structure (high level)

NeuroLikeLab/
├─ app.py
├─ core/
├─ experiments/
├─ datasets/
├─ memory/
├─ prompts/
├─ runs/
└─ logs/   (optional)

Notes

Experimental evidence is stored under runs/ (metrics JSON + JSONL logs).

Avoid committing large artifacts (e.g., JSONL logs). Keep snapshot metrics and summarize key results in README.


---

## これを反映する手順（最短）
1) ローカルの `README.md` を **全消し→上の内容を全貼り→保存**
2) コミットして push

```powershell
git add README.md
git commit -m "Polish README (portfolio-ready)"
git push


