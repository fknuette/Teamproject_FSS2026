# Teamproject FSS2026 – LLM Agents

Dieses Projekt trainiert ein Sprachmodell durch Self-Play in TextArena:

1. Ein Modell spielt mehrere Rollen in `SecretMafia-v0`.
2. Prompts, Antworten, Aktionen und Rewards werden als JSONL gespeichert.
3. Das Modell wird mit GRPO und LoRA auf den erzeugten Daten trainiert.
4. Der LoRA-Adapter wird für den nächsten Rollout in ein vollständiges Modell gemergt.
5. Checkpoints können anschließend gegen ein Basismodell oder mit TrueSkill evaluiert werden.

Die folgenden Befehle und Optionen entsprechen den aktuellen CLI-Parsern im Verzeichnis `scripts/`.

## Voraussetzungen und Installation

Das Projekt verlangt Python 3.13 oder neuer und eine CUDA-fähige GPU. Für das verwendete Modell muss genügend GPU-Speicher verfügbar sein.

Installation mit `uv`:

```bash
uv sync
```

Alle weiteren Befehle werden aus dem Projektverzeichnis ausgeführt.

## Komplette Online-GRPO-Pipeline

Der Online-Loop führt pro Iteration Self-Play, Datensatzerzeugung, GRPO-/LoRA-Training und das Mergen des neuen Modells aus:

```bash
uv run python scripts/online_grpo_loop.py \
  --env-id SecretMafia-v0 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --num-players 8 \
  --num-mafia 2 \
  --loop-count 3 \
  --games-per-iter 3 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --work-dir runs/online_grpo \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --bf16
```

Die wichtigsten Ergebnisse liegen anschließend unter:

```text
runs/online_grpo/
├── traces/                 # Traces jeder Iteration
├── datasets/               # Einzelne und zusammengeführte Datensätze
└── checkpoints/
    └── iter_<N>/
        ├── lora_adapter/   # Trainierter Adapter
        └── merged_model/   # Modell für den nächsten Rollout
```

Wichtige Optionen:

| Option | Standardwert | Bedeutung |
|---|---:|---|
| `--env-id` | `SecretMafia-v0` | TextArena-Environment |
| `--base-model` | `Qwen/Qwen2.5-7B-Instruct` | Ausgangsmodell |
| `--num-players` | `8` | Spielerzahl, erlaubt sind 6 bis 15 |
| `--num-mafia` | `2` | Anzahl der Mafia-Spieler |
| `--loop-count` | `3` | Anzahl der Rollout-/Trainingsiterationen |
| `--games-per-iter` | `3` | Spiele pro Iteration |
| `--tensor-parallel-size` | `1` | Anzahl der GPUs für vLLM Tensor Parallelism |
| `--gpu-memory-utilization` | `0.3` | Von vLLM verwendeter Anteil des GPU-Speichers |
| `--work-dir` | `runs/online_grpo` | Ausgabeordner |

Alle verfügbaren Optionen zeigt:

```bash
uv run python scripts/online_grpo_loop.py --help
```

## Nur Self-Play ausführen

Self-Play verwendet ein gemeinsames Modell für alle Spielerrollen:

```bash
uv run python scripts/self_play_textarena.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --env-id SecretMafia-v0 \
  --num-games 3 \
  --num-players 8 \
  --num-mafia 2 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --output data/selfplay_traces.jsonl
```

`--model` ist hierbei verpflichtend. Die Ausgabedatei wird beim Start geleert und danach mit den Turn-Datensätzen der Spiele befüllt.

Alle verfügbaren Optionen zeigt:

```bash
uv run python scripts/self_play_textarena.py --help
```

## Checkpoints evaluieren

### Ein Checkpoint gegen das Basismodell

Ohne `--eval-checkpoint` wird automatisch der neueste verwendbare Ordner `iter_*` aus dem Checkpoint-Verzeichnis ausgewählt:

```bash
uv run python scripts/evaluation/eval_main.py \
  --mode simple \
  --checkpoint-dir runs/online_grpo/checkpoints \
  --baseline-checkpoint Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3
```

Ein bestimmter Checkpoint wird über seine ID ausgewählt, beispielsweise:

```bash
uv run python scripts/evaluation/eval_main.py \
  --mode simple \
  --checkpoint-dir runs/online_grpo/checkpoints \
  --eval-checkpoint iter_2
```

Das Evaluationsskript bevorzugt `<checkpoint-dir>/<ID>/merged_model`. Falls dieser Ordner nicht existiert, verwendet es `<checkpoint-dir>/<ID>/lora_adapter/final`.

### TrueSkill-Evaluation

```bash
uv run python scripts/evaluation/eval_main.py \
  --mode trueskill \
  --checkpoint-dir runs/online_grpo/checkpoints \
  --baseline-checkpoint Qwen/Qwen2.5-7B-Instruct \
  --min-games-per-team-role 3
```

Standardmäßig wird die bestehende TrueSkill-Registry zu Beginn zurückgesetzt. Um bestehende Ratings weiterzuverwenden:

```bash
uv run python scripts/evaluation/eval_main.py \
  --mode trueskill \
  --checkpoint-dir runs/online_grpo/checkpoints \
  --no-reset-registry
```

Mit `--full` werden im TrueSkill-Modus alle neu entdeckten Checkpoints ausgewertet.

```bash
uv run python scripts/evaluation/eval_main.py \
  --mode trueskill \
  --checkpoint-dir runs/online_grpo/checkpoints \
  --full
```

Die Ergebnisse werden standardmäßig unter `runs/online_grpo/evals/<mode>/results.jsonl` gespeichert.

## Training separat ausführen

Die Trainingslogik befindet sich in `scripts/grpo_training/cli.py` und wird vom Online-Loop direkt über `run_training()` aufgerufen. Die Datei definiert zwar eine `main()`-Funktion, ruft sie aktuell aber nicht über einen `__main__`-Block auf. Deshalb startet

```bash
uv run python scripts/grpo_training/cli.py
```

derzeit kein Training. Für einen vollständigen Lauf sollte `scripts/online_grpo_loop.py` verwendet werden.

## Dynamische Szenario-Loops in `scenarios/`

Die Ordner- und Lauflogik unter `scenarios/` ist der eigentliche "Live-Scenario"-Pfad für dynamische Selbstspiel-Training. Hier werden Rollen, Abstimmungs- und Verteidigungs-Situationen aus echten Spielen geerntet, mit Completionen bewertet und anschließend per GRPO trainiert. Das sind die wichtigsten Einstiegspunkte:

1. `scenarios/dynamic_vote_loop.py`: Nur Abstimmungs-Szenarien aus live Self-Play
2. `scenarios/dynamic_defense_loop.py`: Nur Verteidigungs-/Diskussions-Szenarien mit Judging
3. `scenarios/dynamic_joint_loop.py`: Gemeinsamer Loop für Vote + Defense mit kombiniertem Loss

Diese Loops werden normalerweise aus dem Projektstamm gestartet, nicht über die Root-CLI in `scripts/`.

### 1) Dynamischer Vote-Loop

```bash
uv run python scenarios/dynamic_vote_loop.py \
  --env-id SecretMafia-v0 \
  --num-players 8 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --work-dir scenarios/runs/dynamic_vote_loop \
  --tmp-obs-dir scenarios/tmp_observation_vote \
  --completions-dir scenarios/completions_vote_dynamic \
  --loop-count 4 \
  --games-per-iter 5 \
  --situations-per-game 4 \
  --num-completions 8 \
  --temperature 0.7 \
  --max-tokens 100 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-prompt-length 1024 \
  --max-completion-length 256 \
  --bf16
```

Der Loop sammelt pro Iteration frische Abstimmungsbeobachtungen aus mehreren kompletten Spielen, erzeugt für jede Situation mehrere Completionen, bewertet die Self-Vote-Rewards und trainiert nur auf den neu geernteten Daten. Die Checkpoints landen unter:

```text
scenarios/runs/dynamic_vote_loop/
├── data/
├── checkpoints/
├── tmp_observation_vote/
└── completions_vote_dynamic/
```

### 2) Dynamischer Defense-Loop

```bash
uv run python scenarios/dynamic_defense_loop.py \
  --env-id SecretMafia-v0 \
  --num-players 8 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --judge-mode local \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --work-dir scenarios/runs/dynamic_defense_loop \
  --completions-dir scenarios/completions_defense_dynamic \
  --loop-count 4 \
  --games-per-iter 5 \
  --situations-per-game 4 \
  --num-completions 8 \
  --temperature 0.7 \
  --max-tokens 200 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-prompt-length 1024 \
  --max-completion-length 256 \
  --bf16
```

Dieser Loop sammelt Diskussions-/Verteidigungs-Situationen aus Self-Play, lässt drei andere Villager-Urteile (oder einen Mock-Judge) über die Suspicion vor und nach jeder Completion laufen und berechnet dann team-aware Rewards. Das ist der passende Loop, wenn man speziell auf defensive Spielzüge trainieren will.

Wichtige Ausgabepfade:

```text
scenarios/runs/dynamic_defense_loop/
├── raw_observations/
├── observations/
├── data/
├── checkpoints/
└── completions_defense_dynamic/
```

### 3) Dynamischer Joint-Loop (Vote + Defense)

```bash
uv run python scenarios/dynamic_joint_loop.py \
  --env-id SecretMafia-v0 \
  --num-players 8 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --judge-mode local \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --work-dir scenarios/runs/dynamic_joint_loop \
  --loop-count 4 \
  --games-per-iter 5 \
  --situations-per-game 4 \
  --num-completions 8 \
  --temperature 0.7 \
  --max-tokens 200 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --lambda-vote 1.0 \
  --lambda-defense 1.0 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-prompt-length 1024 \
  --max-completion-length 256 \
  --bf16
```

Das ist der vollständigste Szenario-Loop: In jeder Iteration werden gleichzeitig Abstimmungs- und Verteidigungs-Situation aus denselben Spielen geerntet, beide werden unabhängig bewertet und der gemeinsame GRPO-Loss berücksichtigt beide Ziele in einem Trainingsschritt.

Typische Ausgabeverzeichnisse:

```text
scenarios/runs/dynamic_joint_loop/
├── raw_vote/
├── raw_defense/
├── completions_vote/
├── completions_defense/
├── data/
├── checkpoints/
└── logs/
```

### Wann welche Datei verwenden?

- `dynamic_vote_loop.py`: nur für Abstimmungs-/Votingsituationen
- `dynamic_defense_loop.py`: nur für Diskussions-/Verteidigungsantworten und Suspicion-Scoring
- `dynamic_joint_loop.py`: für die komplette, gemischte Strategie mit Vote + Defense

Die Loop-Parameter `--games-per-iter`, `--situations-per-game` und `--num-completions` steuern den Umfang der Ernte und die Kosten pro Iteration. Mehr Spiele und mehr Situationen erhöhen die Datenvielfalt, aber auch die Laufzeit und den GPU-Aufwand.

## Relevante Projektdateien

- `scripts/argument_parser.py`: zentrale CLI-Argumente und Standardwerte
- `scripts/self_play_textarena.py`: Self-Play und Trace-Erzeugung
- `scripts/online_grpo_loop.py`: vollständiger Rollout-/Trainingsloop
- `scripts/grpo_training/`: Dataset, Modelle, Loss und GRPO-Training
- `scripts/evaluation/`: einfache und TrueSkill-basierte Evaluation
- `src/teamproject_fss2026/textarena_utils.py`: TextArena-Hilfsfunktionen
- `scenarios/dynamic_vote_loop.py`: dynamischer Vote-Loop
- `scenarios/dynamic_defense_loop.py`: dynamischer Defense-Loop
- `scenarios/dynamic_joint_loop.py`: kombinierter Vote+Defense-Loop

## Hinweise

- Die Spielerzahl muss zwischen 6 und 15 liegen.
- Neben der Mafia müssen mindestens ein Doctor und ein Detective Platz haben. Deshalb muss `--num-mafia` zwischen 1 und `num_players - 2` liegen.
- Ein größerer Wert für `--tensor-parallel-size` verteilt vLLM auf mehrere GPUs.
- Falls vLLM beim Start zu viel Speicher reserviert, kann `--gpu-memory-utilization` reduziert werden.
- Das Standardmodell mit 7 Milliarden Parametern benötigt deutlich mehr GPU-Speicher als kleinere Qwen-Varianten.
