"""Berechnet semantische Metriken aus Mafia-JSONL-Traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# Share target parsing with the reader for day votes and Mafia night actions.
from game_reader import (
    _action_outcomes,
    _complete_roles,
    _game_summary,
    _is_vote,
    _night_blocks,
    _roles_from_trace,
    _target,
    _vote_target,
)


METRIC_NAMES = (
    "mafia_self_votes",
    "mafia_vs_mafia_votes",
    "villager_self_votes",
    "mafia_night_consensus",
)


COUNT_METRIC_NAMES = ("mafia_night_friendly_fire", "invalid_moves", "invalid_move_eliminations")

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
    outcomes = _action_outcomes(rows)
    roles = _complete_roles(_roles_from_trace(rows))
    blocks = _night_blocks(rows)
    winner = _game_summary(rows, blocks, roles)
    mafia_players = team_players(roles, "Mafia")
    village_players = team_players(roles, "Village")

    mafia_votes: list[tuple[int, int]] = []
    villager_votes: list[tuple[int, int]] = []
    for row in rows:
        if not _is_vote(row):
            continue

        voter = row.get("player_id")
        target = None if outcomes[id(row)]["invalid"] else outcomes[id(row)]["target"]
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
        for _, day_rows in blocks
    )
    night_rounds = len(blocks)
    consensus_rounds = 0
    unanimous_rounds = 0
    for night_rows, _ in blocks:
        mafia_targets = [
            target
            for row in night_rows
            if row.get("player_id") in mafia_players
            for target in [outcomes[id(row)]["target"]]
            if target is not None and not outcomes[id(row)]["invalid"]
        ]
        if len(mafia_targets) < 2:
            continue
        consensus_rounds += 1
        if len(set(mafia_targets)) == 1:
            unanimous_rounds += 1

    return {
        "game_id": game_id,
        "winner": winner,
        "night_rounds": night_rounds,
        "mafia_consensus_eligible_nights": consensus_rounds,
        "voting_rounds": voting_rounds,
        "mafia_players": len(mafia_players),
        "metrics": {
            "mafia_night_friendly_fire": sum(
                row.get("player_id") in mafia_players
                and outcomes[id(row)]["target"] in mafia_players
                and not outcomes[id(row)]["invalid"]
                for night_rows, _ in blocks for row in night_rows
            ),
            "invalid_moves": sum(outcome["invalid"] for outcome in outcomes.values()),
            "invalid_move_eliminations": sum(outcome["eliminated"] for outcome in outcomes.values()),
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
        "mafia_wins": sum(item["winner"] == "Mafia" for item in game_results),
        "village_wins": sum(item["winner"] == "Village" for item in game_results),
        "metrics": {},
    }
    for name in METRIC_NAMES:
        count = sum(item["metrics"][name]["count"] for item in game_results)
        total = sum(item["metrics"][name]["total"] for item in game_results)
        result["metrics"][name] = metric(count, total)
    for name in COUNT_METRIC_NAMES:
        result["metrics"][name] = sum(item["metrics"][name] for item in game_results)
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


def input_files(input_path: Path) -> list[tuple[str, Path]]:
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Eingabepfad nicht gefunden: {input_path}")

    if not paths:
        raise FileNotFoundError(f"Keine .jsonl-Dateien in {input_path}")
    return [(path.stem, path) for path in paths]


def format_metric(item: dict[str, int | float] | int) -> str:
    if isinstance(item, int):
        return str(item)
    return f"{item['count']} / {item['total']} ({item['rate'] * 100:.2f}%)"


METRIC_LABELS = {
    "mafia_night_friendly_fire": "Mafia-Nachtstimmen auf Mafia (inkl. Selbstziel)",
    "invalid_moves": "Ungültige Aktionen (absolut)",
    "invalid_move_eliminations": "Ausscheiden durch ungültige Aktionen (absolut)",
    "mafia_self_votes": "Mafia-Selbstvotes",
    "mafia_vs_mafia_votes": "Mafia gegen Mafia",
    "villager_self_votes": "Villager-Selbstvotes",
    "mafia_night_consensus": "Mafia-Nachtkonsens",
}


METRIC_GROUPS = (
    ("Voting", ("mafia_vs_mafia_votes", "mafia_self_votes", "villager_self_votes")),
    ("Nachtphase", ("mafia_night_consensus", "mafia_night_friendly_fire")),
    ("Ungültige Aktionen", ("invalid_moves", "invalid_move_eliminations")),
)


def print_metric_groups(metrics: dict[str, Any]) -> None:
    for heading, names in METRIC_GROUPS:
        print(f"    {heading}:")
        for name in names:
            print(f"      {METRIC_LABELS[name]}: {format_metric(metrics[name])}")


def print_game_result(game: dict[str, Any]) -> None:
    print(f"  Spiel {game['game_id']}:")
    print(f"    Gewinner: {game['winner']}")
    print(f"    Nachtrunden: {game['night_rounds']}")
    print(f"    Voting-Runden: {game['voting_rounds']}")
    print_metric_groups(game["metrics"])


def print_report(label: str, report: dict[str, Any]) -> None:
    total = report["aggregate"]
    print(f"\n{label} ({total['games']} Spiele)")
    print("  Gesamt:")
    print(f"    Mafia-Siege: {total['mafia_wins']}")
    print(f"    Village-Siege: {total['village_wins']}")
    print(f"    Nachtrunden insgesamt: {total['night_rounds']}")
    print(f"    Voting-Runden: {total['voting_rounds']}")
    print_metric_groups(total["metrics"])
    for game in report["games"]:
        print_game_result(game)


def markdown_report(reports: dict[str, Any]) -> str:
    lines = ["# Mafia-Metriken", ""]
    for label, report in reports.items():
        total = report["aggregate"]
        lines.extend([f"## {label}", "", "### Gesamt", "",
            f"- Spiele: {total['games']}",
            f"- Mafia-Siege: {total['mafia_wins']}",
            f"- Village-Siege: {total['village_wins']}",
            f"- Nachtrunden insgesamt: {total['night_rounds']}",
            f"- Voting-Runden: {total['voting_rounds']}",
        ])
        for heading, names in METRIC_GROUPS:
            lines.extend(["", f"#### {heading}", ""])
            for name in names:
                lines.append(f"- {METRIC_LABELS[name]}: {format_metric(total['metrics'][name])}")

        lines.extend(["", "### Einzelspiele", "",
            "| Spiel | Gewinner | Nachtrunden | Voting-Runden |",
            "|---:|---|---:|---:|",
        ])
        for game in report["games"]:
            lines.append(
                f"| {game['game_id']} | {game['winner']} | {game['night_rounds']} | {game['voting_rounds']} |"
            )
        for heading, names in METRIC_GROUPS:
            lines.extend(["", f"#### {heading}", "",
                "| Spiel | " + " | ".join(METRIC_LABELS[name] for name in names) + " |",
                "|---:|" + "---:|" * len(names),
            ])
            for game in report["games"]:
                lines.append(f"| {game['game_id']} | " + " | ".join(
                    format_metric(game['metrics'][name]) for name in names
                ) + " |")
        lines.append("")
    lines.extend([
        "Selbstvotes beziehen sich auf das Tagesvoting. Mafia gegen Mafia beim Voting enthält auch Selbstvotes. "
        "Mafia-Nachtstimmen auf lebende Mafia enthalten Selbstziele und zählen Zielwahlen, keine tatsächlichen Kills.", "",
        "Ungültige Aktionen und Ausschlüsse werden pro Zug nach den TextArena-Standardregeln rekonstruiert "
        "(zweiter ungültiger Versuch führt zum Ausscheiden), nicht anhand wiederholter Meldungen gezählt. "
        "Diese beiden Zähler gelten für alle Phasen.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Berechnet semantische Mafia-Spielmetriken aus JSONL-Traces."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Alle Iterationen analysieren")
    mode.add_argument("--iteration", type=int, metavar="N", help="Nur iter_N analysieren")
    mode.add_argument(
        "--input",
        "--file",
        dest="input_path",
        type=Path,
        metavar="PFAD",
        help="Eine einzelne JSONL-Datei oder einen Ordner mit JSONL-Dateien analysieren",
    )
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
        help="Markdown-Ausgabedatei; standardmaessig neben der Eingabe",
    )
    args = parser.parse_args()

    if args.input_path:
        files = input_files(args.input_path)
        default_output = args.input_path.parent / "mafia_metrics.json" if args.input_path.is_file() else args.input_path / "mafia_metrics.json"
    else:
        files = [(f"iter_{iteration}", path) for iteration, path in iteration_files(args.traces_dir, args.iteration)]
        default_output = args.traces_dir / "mafia_metrics.json"

    reports: dict[str, Any] = {}
    for label, trace_path in files:
        games = load_games(trace_path)
        game_results = [
            analyze_game(game_id, rows)
            for game_id, rows in sorted(games.items())
        ]
        report = {"games": game_results, "aggregate": aggregate(game_results)}
        reports[label] = report
        print_report(label, report)

    output_path = args.output or default_output.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_report(reports), encoding="utf-8")
    print(f"\nMarkdown gespeichert: {output_path}")


if __name__ == "__main__":
    main()
