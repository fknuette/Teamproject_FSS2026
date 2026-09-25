"""Assign defense rewards using three local or mocked villager judges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defense_judge import Evaluator, evaluate_suspicion, make_evaluator


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
        baseline = record.get("verdacht_pre")
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not 1 <= baseline <= 10:
            raise ValueError(f"Record {index} has no valid verdacht_pre (1-10)")

    for record in records:
        post = evaluate_suspicion(record["observation"], record["response"], evaluator)
        record["verdacht_post"] = post
        record["reward"] = record["verdacht_pre"] - post

    out_path = Path(output_path) if output_path else path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
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
