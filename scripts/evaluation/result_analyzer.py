"""Simple evaluation result analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "runs"
    / "online_grpo"
    / "evals"
    / "iter_1_vs_base"
    / "results.jsonl"
)

#For Simple MAtchmaking the Winrate per Role is equal to the Winrate per Team
def evaluate_simple_matchmaking_winrate(results_path: str | Path) -> str:
    """Return a readable winrate report for SimplePairMatchmaker results.

    This expects each team in each game to contain exactly one checkpoint.
    Each team-game counts once, even if the team contains multiple players.
    """
    results = _load_jsonl(Path(results_path))
    stats: dict[str, dict[str, Any]] = {}

    for result in results:
        winning_team = result["winning_team"]
        players_by_team = _group_players_by_team(result["players"])

        for team, players in players_by_team.items():
            checkpoints = {player["checkpoint"] for player in players}
            if len(checkpoints) != 1:
                raise ValueError(
                    "Simple matchmaking winrate expects one checkpoint per team. "
                    f"Game {result.get('game_id')} team {team} has: "
                    f"{sorted(checkpoints)}"
                )

            model = next(iter(checkpoints))
            won = team == winning_team
            model_stats = _get_model_stats(stats, model)

            _add_result(model_stats["overall"], won)
            _add_result(model_stats["by_team"].setdefault(team, _empty_counts()), won)

            for role in sorted({player["role"] for player in players}):
                _add_result(
                    model_stats["by_role"].setdefault(role, _empty_counts()),
                    won,
                )

    return _render_winrate_report(total_games=len(results), stats=stats)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    results = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return results


def _group_players_by_team(players: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    players_by_team: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        players_by_team.setdefault(player["team"], []).append(player)
    return players_by_team


def _get_model_stats(stats: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    return stats.setdefault(
        model,
        {
            "overall": _empty_counts(),
            "by_team": {},
            "by_role": {},
        },
    )


def _empty_counts() -> dict[str, int]:
    return {"wins": 0, "losses": 0}


def _add_result(counts: dict[str, int], won: bool) -> None:
    if won:
        counts["wins"] += 1
    else:
        counts["losses"] += 1


def _render_winrate_report(total_games: int, stats: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"Total Games: {total_games}",
        f"Models: {len(stats)}",
        "",
    ]

    for model in sorted(stats):
        model_stats = stats[model]
        lines.extend(
            [
                "=" * 46,
                f"Model: {_display_model_name(model)}",
                "=" * 46,
                "",
                "Overall",
                f"  {_format_counts(model_stats['overall'])}",
                "",
                "By Team",
            ]
        )

        for team in sorted(model_stats["by_team"]):
            lines.append(
                f"  {team:<10} {_format_counts(model_stats['by_team'][team])}"
            )

        lines.extend(["", "By Role"])
        for role in sorted(model_stats["by_role"], key=_role_sort_key):
            lines.append(
                f"  {role:<10} {_format_counts(model_stats['by_role'][role])}"
            )

        lines.extend(["", ""])

    return "\n".join(lines).rstrip()


def _format_counts(counts: dict[str, int]) -> str:
    wins = counts["wins"]
    losses = counts["losses"]
    games = wins + losses
    winrate = wins / games if games else 0.0
    return (
        f"Winrate: {winrate * 100:6.2f}%   "
        f"Games: {games:<3} Wins: {wins:<3} Losses: {losses}"
    )


def _display_model_name(model: str) -> str:
    marker = "/scripts/runs/online_grpo/checkpoints/"
    if marker in model:
        return model.split(marker, maxsplit=1)[1]
    return model


def _role_sort_key(role: str) -> tuple[int, str]:
    preferred_order = {"werewolf": 0, "villager": 1}
    return preferred_order.get(role, 100), role


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SimplePairMatchmaker winrates from results.jsonl."
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to results.jsonl (default: {DEFAULT_RESULTS_PATH})",
    )
    args = parser.parse_args()

    print(evaluate_simple_matchmaking_winrate(args.results_path))


if __name__ == "__main__":
    main()
