# 【JP NOTE】これは Claude Code 向けの指示書です（人間向けではありません）
# 目的：ファイルを壊さず、最小diffで、再現可能な研究プロトタイプを維持すること。
# 重要：勝手なリファクタ・構成変更・ファイル削除/改名は禁止。必ず diff-only / minimal changes.

# NeuroLikeLab - Claude Code Instructions (v0.3 / Phase3-ready)

## Goal
Build a minimal, reproducible prototype for decision/persona experiments:

1) `/analyze` (v0.2 core)
text -> emotion -> latent_state -> state_update -> JSON response + JSONL logging

2) `/persona` (Phase3 extension)
`/analyze`-equivalent state update + call Ollama (local LLM) -> persona JSON response  
+ JSONL logging in `runs/`

3) `experiments/run_eval.py` (v0.3)
Run a fixed JSONL dataset through `/persona` and save metrics (accuracy + distributions).

## Non-goals (strict)
- Do NOT add UI, DB, auth, Docker, or extra features.
- Do NOT introduce new frameworks beyond FastAPI/Pydantic/httpx/requests.
- Do NOT over-engineer: keep it small, stable, inspectable.
- Do NOT add agent frameworks (LangChain, etc.).
- Do NOT refactor structure unless explicitly requested.

## Constraints (strict)
- Port is always 8010 (8000 is used by another project).
- Do NOT delete/rename existing files without explicit instruction.
- Prefer additive changes.
- If a file must change, keep the diff minimal.
- All logs must be UTF-8 and readable with `Get-Content -Encoding utf8`.

## Project structure (must exist)
- app.py (FastAPI)
- requirements.txt
- data/modulator_profiles.json
- core/mix.py
- core/state.py
- core/io.py
- logs/ (created at runtime)
- experiments/
  - eval_20cases.jsonl
  - run_eval.py
- runs/ (created at runtime)
- README.md

## API Contracts

### /health (must match)
Returns:
```json
{"ok": true, "version": "v0.2"}
