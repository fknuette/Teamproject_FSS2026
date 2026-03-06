# Teamproject FSS2026 LLM-Agents

MVP-Pipeline für euer Projekt:
- Zwei LLMs spielen gegeneinander in TextArena (Self-Play, über `vLLM`)
- Pro Zug werden Prompt, Reasoning-Trace und Action gespeichert
- Nach jedem Match wird Win/Loss als Reward vergeben (`1` für Gewinner, `0` für Verlierer)
- Danach LoRA-Finetuning mit GRPO (`trl`) auf diesen Traces

## 1) Installation

```bash
uv sync
```

Falls ihr kein `uv` nutzt:

```bash
pip install -e .
```

## 2) Self-Play Daten sammeln

Beispiel mit zwei Qwen-Modellen in TicTacToe:

```bash
python scripts/self_play_textarena.py \
	--env-id TicTacToe-v0 \
	--model-a Qwen/Qwen2.5-3B-Instruct \
	--model-b Qwen/Qwen2.5-3B-Instruct \
	--num-games 50 \
	--output data/selfplay_traces.jsonl
```

Output-Datei: `data/selfplay_traces.jsonl`

Jede Zeile enthält u. a.:
- `prompt`
- `reasoning_trace`
- `action`
- `final_reward` (1/0 bzw. env-reward)
- `won` (1/0)

## 3) LoRA + GRPO Training

```bash
python scripts/train_grpo_lora.py \
	--model Qwen/Qwen2.5-3B-Instruct \
	--data data/selfplay_traces.jsonl \
	--output-dir outputs/grpo-lora \
	--epochs 1 \
	--batch-size 2 \
	--gradient-accumulation-steps 8 \
	--bf16
```

Gespeichert wird das LoRA-Checkpoint unter `outputs/grpo-lora`.

## 3b) Ein einziger Online-Loop (ohne manuelles Neustarten)

Das folgende Skript macht automatisch pro Iteration:
1. Rollout (LLM vs LLM in TextArena)
2. Reward/Trace-Dataset aktualisieren
3. GRPO + LoRA trainieren
4. LoRA in ein vollständiges Modell mergen (für nächste vLLM-Runde)

```bash
python scripts/online_grpo_loop.py \
	--env-id TicTacToe-v0 \
	--base-model Qwen/Qwen2.5-3B-Instruct \
	--loop-count 3 \
	--games-per-iter 30 \
	--work-dir runs/online_grpo \
	--epochs 1 \
	--batch-size 2 \
	--gradient-accumulation-steps 8 \
	--bf16
```

Optional:
- `--opponent-model ...` für festen Gegner
- `--skip-merge` falls ihr nur Adapter trainieren wollt (dann keine Policy-Aktualisierung für den nächsten Rollout)

## 4) Wichtige Projektdateien

- `scripts/self_play_textarena.py`: vLLM Self-Play + Trace-Logging
- `scripts/train_grpo_lora.py`: GRPO-Training mit LoRA
- `scripts/online_grpo_loop.py`: durchgehender Rollout→Training-Loop
- `src/teamproject_fss2026/textarena_utils.py`: Prompt- und Parsing-Utilities

## Hinweise

- Für echte große Modelle braucht ihr genügend VRAM/GPU.
- Startet zuerst mit kleinen Environments (`TicTacToe-v0`) und kleinen Modellen.
- Wenn ihr SecretMafia trainieren wollt, erhöht später schrittweise Komplexität und Match-Anzahl.