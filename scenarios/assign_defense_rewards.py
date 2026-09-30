"""Assign defense rewards using three local or mocked villager judges.

Two reward functions are provided:
- ``assign_defense_rewards``: legacy single-player reward (verdacht_pre - verdacht_post).
- ``assign_team_aware_rewards``: team-aware multi-player reward for the dynamic loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defense_judge import Evaluator, evaluate_all_suspicions, evaluate_suspicion, make_evaluator


def assign_defense_rewards(
    input_path: str,
    output_path: str = "",
    evaluator: Evaluator | None = None,
) -> list[dict]:
    """Write reward = verdacht_pre - verdacht_post into a completion JSONL."""
    path = Path(input_path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    if evaluator is None:
        evaluator = make_evaluator("mock")

    # Validate the whole file before writing any result.
    for index, record in enumerate(records):
        if not isinstance(record.get("observation"), str) or not isinstance(record.get("response"), str):
            raise ValueError(f"Record {index} needs string observation and response")
        if "judge_observation" in record and not isinstance(record["judge_observation"], str):
            raise ValueError(f"Record {index} has invalid judge_observation")
        baseline = record.get("verdacht_pre")
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not 1 <= baseline <= 10:
            raise ValueError(f"Record {index} has no valid verdacht_pre (1-10)")

    for record in records:
        post = evaluate_suspicion(record.get("judge_observation", record["observation"]), record["response"], evaluator)
        record["verdacht_post"] = post
        record["reward"] = record["verdacht_pre"] - post

    out_path = Path(output_path) if output_path else path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def assign_team_aware_rewards(
    records: list[dict],
    output_path: str | Path = "",
    input_path: str | Path = "",
    evaluator: Evaluator | None = None,
) -> list[dict]:
    """Evaluate post-completion suspicion of all players and compute team-aware rewards.

    Each record must already contain:
        observation, response, player_id, player_team, judge_ids,
        all_player_ids, villager_ids, mafia_ids,
        suspicion_pre (dict[str, float]),
        judge_observation (public-only dialogue string).

    Reward formula:
        Mafia speaker: Σ(post[v]-pre[v] for villagers) + Σ(pre[m]-post[m] for mafia)
        Village speaker: Σ(pre[v]-post[v] for villagers) + Σ(post[m]-pre[m] for mafia)
    """
    if not records:
        raise ValueError("No records provided")
    if evaluator is None:
        evaluator = make_evaluator("mock")

    for index, record in enumerate(records):
        for field in ("observation", "response", "player_team", "judge_observation"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Record {index} missing or invalid field '{field}'")
        for list_field in ("judge_ids", "all_player_ids", "villager_ids", "mafia_ids"):
            if not isinstance(record.get(list_field), list):
                raise ValueError(f"Record {index} missing or invalid field '{list_field}'")
        if not isinstance(record.get("suspicion_pre"), dict):
            raise ValueError(f"Record {index} missing or invalid field 'suspicion_pre'")
        if record["player_team"] not in ("Mafia", "Village"):
            raise ValueError(f"Record {index} has unknown player_team: {record['player_team']!r}")

    for record in records:
        suspicion_post = evaluate_all_suspicions(
            observation=record["observation"],
            public_game_state=record["judge_observation"],
            response=record["response"],
            judge_ids=record["judge_ids"],
            all_player_ids=record["all_player_ids"],
            evaluator=evaluator,
        )
        record["suspicion_post"] = suspicion_post

        pre = {int(k): float(v) for k, v in record["suspicion_pre"].items()}
        post = suspicion_post
        villagers: list[int] = record["villager_ids"]
        mafia: list[int] = record["mafia_ids"]

        if record["player_team"] == "Mafia":
            reward = (
                sum(post[v] - pre[v] for v in villagers)
                + sum(pre[m] - post[m] for m in mafia)
            )
        else:
            reward = (
                sum(pre[v] - post[v] for v in villagers)
                + sum(post[m] - pre[m] for m in mafia)
            )
        record["reward"] = reward

    out_path = Path(output_path) if output_path else (Path(input_path) if input_path else None)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate defense completions with three villager judges")
    parser.add_argument("input_path", help="Completions JSONL with cached verdacht_pre")
    parser.add_argument("--output", default="", help="Output JSONL; default overwrites input")
    parser.add_argument("--judge-mode", choices=("mock", "local"), default="mock")
    parser.add_argument("--judge-model", default="", help="Local model path/HF id for local vLLM evaluation")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    records = assign_defense_rewards(args.input_path, args.output, evaluator)
    print(f"[Saved] rated {len(records)} completions")


if __name__ == "__main__":
    main()
