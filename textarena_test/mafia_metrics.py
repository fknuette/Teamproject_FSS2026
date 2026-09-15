"""Berechnet semantische Metriken aus Mafia-JSONL-Traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from game_reader import (
    _complete_roles,
    _is_vote,
    _night_blocks,
    _roles_from_trace,
    _target,
)


METRIC_NAMES = (
    "mafia_self_votes",
    "mafia_vs_mafia_votes",
    "villager_self_votes",
    "mafia_night_consensus",
)


def load_games(trace_path: Path) -> dict[int, list[dict[str, Any]]]:
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Ungültiges JSON in {trace_path}, Zeile {line_number}"
                ) from error
            games[row["game_id"]].append(row)
    return dict(games)


def percentage(count: int, total: int) -> float:
    return count / total if total else 0.0


def metric(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "total": total, "rate": percentage(count, total)}


def team_players(roles: dict[int, tuple[str, str]], team: str) -> set[int]:
    return {
        player_id
        for player_id, (_, player_team) in roles.items()
        if player_team.lower() == team.lower()
    }


def analyze_game(game_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles = _complete_roles(_roles_from_trace(rows))
    mafia_players = team_players(roles, "Mafia")
    village_players = team_players(roles, "Village")

    mafia_votes: list[tuple[int, int]] = []
    villager_votes: list[tuple[int, int]] = []
    for row in rows:
        if not _is_vote(row):
            continue

        voter = row.get("player_id")
        target = _target(row.get("response", ""))
        if not isinstance(voter, int) or target is None:
            continue

        if voter in mafia_players:
            mafia_votes.append((voter, target))
        elif voter in village_players:
            villager_votes.append((voter, target))

    mafia_self_votes = sum(voter == target for voter, target in mafia_votes)
    mafia_vs_mafia_votes = sum(
        target in mafia_players
        for voter, target in mafia_votes
    )
    villager_self_votes = sum(voter == target for voter, target in villager_votes)

    voting_rounds = sum(
        any(_is_vote(row) for row in day_rows)
        for _, day_rows in _night_blocks(rows)
    )
    night_rounds = len(_night_blocks(rows))
    consensus_rounds = 0
    unanimous_rounds = 0
    for night_rows, _ in _night_blocks(rows):
        mafia_targets = [
            target
            for row in night_rows
            if row.get("player_id") in mafia_players
            for target in [_target(row.get("response", ""))]
            if target is not None
        ]
        if len(mafia_targets) < 2:
            continue
        consensus_rounds += 1
        if len(set(mafia_targets)) == 1:
            unanimous_rounds += 1

    return {
        "game_id": game_id,
        "night_rounds": night_rounds,
        "mafia_consensus_eligible_nights": consensus_rounds,
        "voting_rounds": voting_rounds,
        "mafia_players": len(mafia_players),
        "metrics": {
            "mafia_self_votes": metric(mafia_self_votes, len(mafia_votes)),
            "mafia_vs_mafia_votes": metric(mafia_vs_mafia_votes, len(mafia_votes)),
            "villager_self_votes": metric(
                villager_self_votes, len(villager_votes)
            ),
            "mafia_night_consensus": metric(
                unanimous_rounds, consensus_rounds
            ),
        },
    }


def aggregate(game_results: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "games": len(game_results),
        "night_rounds": sum(item["night_rounds"] for item in game_results),
        "mafia_consensus_eligible_nights": sum(
            item["mafia_consensus_eligible_nights"] for item in game_results
        ),
        "voting_rounds": sum(item["voting_rounds"] for item in game_results),
        "metrics": {},
    }
    for name in METRIC_NAMES:
        count = sum(item["metrics"][name]["count"] for item in game_results)
        total = sum(item["metrics"][name]["total"] for item in game_results)
        result["metrics"][name] = metric(count, total)
    return result


def iteration_files(
    traces_dir: Path, iteration: int | None
) -> list[tuple[int, Path]]:
    if iteration is not None:
        paths = [traces_dir / f"iter_{iteration}.jsonl"]
    else:
        paths = sorted(traces_dir.glob("iter_*.jsonl"))

    if not paths or any(not path.exists() for path in paths):
        wanted = f"iter_{iteration}.jsonl" if iteration is not None else "iter_*.jsonl"
        raise FileNotFoundError(f"Keine {wanted}-Datei in {traces_dir}")

    return [(int(path.stem.split("_")[-1]), path) for path in paths]


def format_metric(item: dict[str, int | float]) -> str:
    return f"{item['count']} / {item['total']} ({item['rate'] * 100:.2f}%)"


def print_report(iteration: int, report: dict[str, Any]) -> None:
    print(f"\nIteration {iteration} ({report['aggregate']['games']} Spiele)")
    print(f"Nachtrunden insgesamt: {report['aggregate']['night_rounds']}")
    print(
        "Nachtrunden mit mindestens zwei aktiven Mafia-Spielern: "
        f"{report['aggregate']['mafia_consensus_eligible_nights']}"
    )
    print(f"Voting-Runden: {report['aggregate']['voting_rounds']}")
    labels = {
        "mafia_self_votes": "Mafia-Selbstvotes",
        "mafia_vs_mafia_votes": "Mafia gegen Mafia",
        "villager_self_votes": "Villager-Selbstvotes",
        "mafia_night_consensus": "Mafia-Nachtkonsens",
    }
    for name in METRIC_NAMES:
        print(f"  {labels[name]}: {format_metric(report['aggregate']['metrics'][name])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Berechnet semantische Mafia-Spielmetriken aus JSONL-Traces."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Alle Iterationen analysieren")
    mode.add_argument("--iteration", type=int, metavar="N", help="Nur iter_N analysieren")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "scripts"
        / "runs"
        / "online_grpo"
        / "traces",
        help="Ordner mit iter_*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON-Ausgabedatei; standardmaessig im traces-Ordner",
    )
    args = parser.parse_args()

    reports: dict[str, Any] = {}
    for iteration, trace_path in iteration_files(args.traces_dir, args.iteration):
        games = load_games(trace_path)
        game_results = [
            analyze_game(game_id, rows)
            for game_id, rows in sorted(games.items())
        ]
        report = {"games": game_results, "aggregate": aggregate(game_results)}
        reports[f"iter_{iteration}"] = report
        print_report(iteration, report)

    output_path = args.output or args.traces_dir / "mafia_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nJSON gespeichert: {output_path}")


if __name__ == "__main__":
    main()
