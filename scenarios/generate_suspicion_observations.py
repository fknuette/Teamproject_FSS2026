"""Create one to three fixed Secret Mafia defense observations with cached baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defense_judge import evaluate_suspicion, make_evaluator

HERE = Path(__file__).resolve().parent

SCENARIOS = [
    {
        "player_id": 2,
        "history": [
            "[Player 5] Spieler 2 wich meiner Frage zur Nachtaktion aus.",
            "[Player 1] Und er wechselte sofort das Thema. Spieler 2, erklär das bitte.",
        ],
    },
    {
        "player_id": 4,
        "history": [
            "[Player 6] Spieler 4 hat gestern stark gegen 0 argumentiert. 0 war Dorfbewohner.",
            "[Player 3] Danach hat 4 seine Einschätzung ohne Begründung geändert.",
            "[Player 7] Spieler 4, wie passt dieser Wechsel zusammen?",
        ],
    },
    {
        "player_id": 6,
        "history": [
            "[Player 0] Spieler 6 hat erst 3 verteidigt und später gegen 3 gestimmt.",
            "[Player 5] Dieser Wechsel kurz vor der Abstimmung wirkt abgesprochen.",
            "[Player 1] Spieler 6, begründe deine Entscheidung vor der nächsten Wahl.",
        ],
    },
]


def build_observation(player_id: int, history: list[str]) -> str:
    """Keep the exact [GAME] Discuss format expected by the existing prompt chain."""
    return (
        f"[GAME] Welcome to Secret Mafia! You are Player {player_id}.\n"
        "Your role is unknown to the other players. During discussions, your public "
        "statement is broadcast to everyone.\n\n"
        "[GAME] Daytime discussion. Public history:\n"
        + "\n".join(history)
        + "\n[GAME] Discuss phase: The players suspect you. Respond to their accusations now."
    )


def generate_observations(
    count: int, output_dir: Path, evaluator, judge_mode: str = "mock", judge_model: str = ""
) -> list[Path]:
    if not 1 <= count <= len(SCENARIOS):
        raise ValueError("count must be between 1 and 3")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, scenario in enumerate(SCENARIOS[:count], start=1):
        observation = build_observation(scenario["player_id"], scenario["history"])
        baseline = evaluate_suspicion(observation, None, evaluator)
        path = output_dir / f"defense_{index}.json"
        data = {
            "player_id": scenario["player_id"],
            "history": scenario["history"],
            "observation": observation,
            "verdacht_pre": baseline,
            "judge_mode": judge_mode,
            "judge_model": judge_model if judge_mode == "local" else "",
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
        print(f"[Saved] {path} (verdacht_pre={baseline:.2f})")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed defense observations and cache pre-ratings")
    parser.add_argument("--count", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=HERE / "observations_defense")
    parser.add_argument("--judge-mode", choices=("mock", "local"), default="mock")
    parser.add_argument("--judge-model", default="", help="Local model path/HF id for local vLLM evaluation")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    generate_observations(args.count, args.output_dir, evaluator, args.judge_mode, args.judge_model)


if __name__ == "__main__":
    main()
