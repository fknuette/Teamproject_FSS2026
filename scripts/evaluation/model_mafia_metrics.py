"""Attribute Mafia metrics to a checkpoint using evaluation results and turns.

Run directly with --model CHECKPOINT_ID. Only traced games contribute action
metrics; win rates include every result. Consensus is a team-context metric,
counted once per eligible night in which the selected players participated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Reuse the same action parsing and metric definitions as mafia_metrics.py.
_READER_DIR = Path(__file__).resolve().parents[2] / "textarena_test"
if str(_READER_DIR) not in sys.path:
    sys.path.insert(0, str(_READER_DIR))
from game_reader import _action_outcomes, _is_vote, _night_blocks, _phase
from mafia_metrics import COUNT_METRIC_NAMES, METRIC_NAMES, metric, METRIC_GROUPS, METRIC_LABELS

PAPER_EXTRA_METRICS = (
    "invalid_action_rate", "invalid_elimination_rate", "mafia_teammate_vote_rate", "night_friendly_fire_rate",
)
PAPER_METRICS = (
    ("invalid_action_rate", "Ungültige Aktionen", "↓"),
    ("invalid_elimination_rate", "Ausscheiden durch ungültige Aktionen", "↓"),
    ("mafia_self_votes", "Mafia-Selbstvotes", "deskriptiv"),
    ("villager_self_votes", "Village-Selbstvotes", "deskriptiv"),
    ("mafia_teammate_vote_rate", "Mafia-Votes auf andere Mafia", "deskriptiv"),
    ("night_friendly_fire_rate", "Mafia-Nachtziele auf eigenes Team", "↓"),
    ("mafia_night_consensus", "Mafia-Nachtkonsens", "deskriptiv"),
)


def team_winrates(entries: list[dict]) -> dict:
    rates = {}
    for team in ("Mafia", "Village"):
        games = [e for e in entries if any(p["team"] == team for p in e["players"])]
        wins = sum(e["winning_team"] == team for e in games)
        rates[team] = metric(wins, len(games))
    return rates


DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "runs/online_grpo/evals/trueskill/results.jsonl"


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}, line {number}")
            rows.append(row)
    return rows


def _player_metrics(rows: list[dict], players: list[dict], selected: set[int]) -> dict:
    roles = {p["player_id"]: (p["role"], p["team"]) for p in players}
    outcomes = _action_outcomes(rows, roles=roles)
    mafia = {pid for pid, (_, team) in roles.items() if team.lower() == "mafia"}
    village = {pid for pid, (_, team) in roles.items() if team.lower() == "village"}
    actions = [r for r in rows if r["player_id"] in selected]
    votes = [r for r in actions if _is_vote(r) and not outcomes[id(r)]["invalid"] and outcomes[id(r)]["target"] is not None]
    mafia_votes = [r for r in votes if r["player_id"] in mafia]
    village_votes = [r for r in votes if r["player_id"] in village]
    eligible = unanimous = friendly_fire = 0
    for night, _ in _night_blocks(rows):
        valid = [r for r in night if r["player_id"] in mafia and not outcomes[id(r)]["invalid"] and outcomes[id(r)]["target"] is not None]
        participating = [r for r in valid if r["player_id"] in selected]
        friendly_fire += sum(outcomes[id(r)]["target"] in mafia for r in participating)
        if participating and len(valid) >= 2:
            eligible += 1
            unanimous += len({outcomes[id(r)]["target"] for r in valid}) == 1
    targeted = [r for r in actions if _is_vote(r) or (
        _phase(r) == "NACHT" and roles[r["player_id"]][0].lower() in {"mafia", "doctor", "detective"}
    )]
    valid_night = [r for night, _ in _night_blocks(rows) for r in night
                   if r["player_id"] in selected & mafia and not outcomes[id(r)]["invalid"]
                   and outcomes[id(r)]["target"] is not None]
    eliminated = {r["player_id"] for r in actions if outcomes[id(r)]["eliminated"]}
    return {
        "invalid_action_rate": metric(sum(outcomes[id(r)]["invalid"] for r in targeted), len(targeted)),
        "invalid_elimination_rate": metric(len(eliminated), len(selected)),
        "mafia_teammate_vote_rate": metric(sum(outcomes[id(r)]["target"] in mafia and outcomes[id(r)]["target"] != r["player_id"] for r in mafia_votes), len(mafia_votes)),
        "night_friendly_fire_rate": metric(friendly_fire, len(valid_night)),
        "mafia_self_votes": metric(sum(r["player_id"] == outcomes[id(r)]["target"] for r in mafia_votes), len(mafia_votes)),
        "mafia_vs_mafia_votes": metric(sum(outcomes[id(r)]["target"] in mafia for r in mafia_votes), len(mafia_votes)),
        "villager_self_votes": metric(sum(r["player_id"] == outcomes[id(r)]["target"] for r in village_votes), len(village_votes)),
        "mafia_night_consensus": metric(unanimous, eligible),
        "mafia_night_friendly_fire": friendly_fire,
        "invalid_moves": sum(outcomes[id(r)]["invalid"] for r in actions),
        "invalid_move_eliminations": sum(outcomes[id(r)]["eliminated"] for r in actions),
    }


def _summarize(entries: list[dict]) -> dict:
    traced = [e for e in entries if e["metrics"] is not None]
    metrics = {name: metric(sum(e["metrics"][name]["count"] for e in traced), sum(e["metrics"][name]["total"] for e in traced)) for name in (*METRIC_NAMES, *PAPER_EXTRA_METRICS)}
    metrics.update({name: sum(e["metrics"][name] for e in traced) for name in COUNT_METRIC_NAMES})
    appearances = sum(len(e["players"]) for e in entries)
    wins = sum(p["team"] == e["winning_team"] for e in entries for p in e["players"])
    return {
        "games": len(entries), "traced_games": len(traced),
        "player_appearances": appearances,
        "winrate": metric(wins, appearances),
        "traced_actions": sum(e["actions"] for e in traced),
        "metrics": metrics,
    }


def analyze_model(results_path: str | Path, model: str, turns_path: str | Path | None = None) -> dict:
    """Join by game_id/player_id; match checkpoint identifiers exactly."""
    results_path = Path(results_path)
    turns_path = Path(turns_path) if turns_path is not None else results_path.with_name(results_path.stem + "_turns.jsonl")
    results = _load(results_path)
    games = {}
    for result in results:
        gid = result["game_id"]
        if gid in games:
            raise ValueError(f"Duplicate game_id in results: {gid}")
        ids = [p["player_id"] for p in result["players"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate player_id in game {gid}")
        games[gid] = result
    turns: dict[int, list[dict]] = {}
    for row in _load(turns_path):
        # Evaluation files also contain end_reason records without actions.
        if "player_id" not in row:
            continue
        gid = row["game_id"]
        if gid in games and row["player_id"] not in {p["player_id"] for p in games[gid]["players"]}:
            raise ValueError(f"Unknown player {row['player_id']} in game {gid}")
        turns.setdefault(gid, []).append(row)
    entries = []
    by_role: dict[str, list[dict]] = {}
    by_team: dict[str, list[dict]] = {}
    for gid, game in games.items():
        selected = [p for p in game["players"] if p["checkpoint"] == model]
        if not selected:
            continue
        rows = turns.get(gid)
        def entry(players: list[dict]) -> dict:
            ids = {p["player_id"] for p in players}
            return {
                "game_id": gid, "winning_team": game["winning_team"], "players": players,
                "actions": sum(r["player_id"] in ids for r in rows) if rows else 0,
                "metrics": _player_metrics(rows, game["players"], ids) if rows else None,
            }
        entries.append(entry(selected))
        for key, groups in [("role", by_role), ("team", by_team)]:
            for value in sorted({p[key] for p in selected}):
                groups.setdefault(value, []).append(entry([p for p in selected if p[key] == value]))
    if not entries:
        available = sorted({p["checkpoint"] for g in results for p in g["players"]})
        raise ValueError(f"No results for checkpoint {model!r}. Available: {', '.join(available)}")
    return {
        "model": model, "results_path": str(results_path), "turns_path": str(turns_path),
        "coverage": {
            "result_games": len(games), "trace_games": len(set(games) & set(turns)),
            "missing_trace_game_ids": sorted(set(games) - set(turns)),
            "orphan_trace_game_ids": sorted(set(turns) - set(games)),
            "model_missing_trace_game_ids": [e["game_id"] for e in entries if e["metrics"] is None],
        },
        "definitions": {
            "action_metrics": "Only actions of selected checkpoint players in games with traces; rates pool counts and denominators.",
            "winrate": "Wins per player appearance across all results, including games without traces.",
            "mafia_night_consensus": "Team unanimity once per night with at least two valid Mafia actions and a valid action from a selected player; includes teammates from other models. Role/team groups are not necessarily additive.",
            "zero_denominator": "A total of zero means no eligible observations, not evidence of a zero event rate.",
            "coverage": "Trace presence does not guarantee that a game's trace is complete.",
        },
        "team_winrates": team_winrates(entries),
        "overall": _summarize(entries),
        "by_role": {k: _summarize(v) for k, v in by_role.items()},
        "by_team": {k: _summarize(v) for k, v in by_team.items()},
        "games": entries,
    }


def markdown_report(reports: list[dict], baseline: str | None = None) -> str:
    """Compact scientific report: coverage, game-level wins, action rates."""
    def rate(value: dict) -> str:
        return f"{value['rate']:.1%} ({value['count']}/{value['total']})" if value["total"] else "—"

    lines = ["# Modellvergleich: Mafia-Evaluation", "", "## Datengrundlage", ""]
    roles = sorted({p["role"] for r in reports for g in r["games"] for p in g["players"]})
    lines += ["| Modell | Spiele | Mit Traces | " + " | ".join(roles) + " |",
              "|---|---:|---:|" + "---:|" * len(roles)]
    for r in reports:
        counts = [sum(p["role"] == role for g in r["games"] for p in g["players"]) for role in roles]
        lines.append(f"| {r['model']} | {r['overall']['games']} | {r['overall']['traced_games']} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "Rollen zählen Spieler-Einsätze aus allen Ergebnissen; mehrere Spieler desselben Modells können im selben Spiel auftreten.", ""]
    for r in reports:
        missing = r["coverage"]["model_missing_trace_game_ids"]
        if missing:
            lines.append(f"- **{r['model']}**: fehlende Turn-Traces für Spiele {', '.join(map(str, missing))}.")
        orphan = r["coverage"]["orphan_trace_game_ids"]
        if orphan:
            lines.append(f"- **{r['model']}**: Traces ohne Ergebnis ausgeschlossen: {orphan}.")
    lines += ["", "## Spielerfolg", "", "Winrate: Prozent (Siege/Team-Spiele).", "",
              "| Modell | Als Mafia | Als Village |", "|---|---:|---:|"]
    for r in reports:
        lines.append(f"| {r['model']} | {rate(r['team_winrates']['Mafia'])} | {rate(r['team_winrates']['Village'])} |")
    reference = next((r for r in reports if r["model"] == baseline), None)
    if reference and len(reports) > 1:
        lines += ["", f"**Differenz zum Basismodell `{baseline}`**, in Prozentpunkten (deskriptiv):", "",
                  "| Modell | Δ Mafia | Δ Village |", "|---|---:|---:|"]
        for r in reports:
            if r is reference:
                continue
            differences = []
            for team in ("Mafia", "Village"):
                v, ref = r["team_winrates"][team], reference["team_winrates"][team]
                differences.append(f"{100 * (v['rate'] - ref['rate']):+.1f}" if v["total"] and ref["total"] else "—")
            lines.append(f"| {r['model']} | " + " | ".join(differences) + " |")
    lines += ["", "Jedes Spiel zählt pro Modell und Team einmal, unabhängig von der Zahl eigener Spieler. Ist ein Modell auf beiden Seiten vertreten, trägt das Spiel zu beiden Team-Spalten bei. Baseline-Differenzen sind deskriptiv; es wird kein Signifikanztest durchgeführt.", "",
              "## Verhalten", "", "Prozent (Ereignisse/Gelegenheiten), nur aus Spielen mit Turn-Traces. ↑/↓ kennzeichnet die gewünschte Richtung; deskriptive Werte sind kein eindeutiges Qualitätsurteil.", "",
              "| Metrik | Richtung | " + " | ".join(r["model"] for r in reports) + " |",
              "|---|---|" + "---:|" * len(reports)]
    for name, label, direction in PAPER_METRICS:
        lines.append(f"| {label} | {direction} | " + " | ".join(rate(r["overall"]["metrics"][name]) for r in reports) + " |")
    lines += ["", "### Nenner und Interpretation", "",
              "- **Ungültige Aktionen:** alle zielpflichtigen Aktionen, einschließlich ungültiger Versuche und Wiederholungen. Freie Diskussion ist ausgeschlossen.",
              "- **Ausscheiden durch ungültige Aktionen:** betroffene Modellspieler / Spieler-Einsätze in Spielen mit Traces; jeder Spieler zählt höchstens einmal.",
              "- **Mafia-Selbstvotes:** Selbstziele / gültige Mafia-Tagesvotes.",
              "- **Village-Selbstvotes:** Selbstziele / gültige Tagesvotes aller Village-Rollen, einschließlich Doctor und Detective.",
              "- **Mafia-Votes auf andere Mafia:** Mitspieler-Ziele ohne Selbstvotes / gültige Mafia-Tagesvotes. Kann der Tarnung dienen.",
              "- **Mafia-Nachtziele auf eigenes Team:** eigene Mafia-Ziele einschließlich Selbstziel / gültige Mafia-Nachtaktionen.",
              "- **Nachtkonsens:** einstimmige / auswertbare Nächte mit mindestens zwei gültigen Mafia-Aktionen und gültiger Modellbeteiligung. Bezieht auch andere Modelle im Team ein; Einigkeit garantiert keine gute Zielwahl.",
              "", "## Aussagegrenzen", "",
              "Die Ergebnisse beschreiben die vorhandenen Matchups. Unterschiedliche Gegner, Mitspieler und Rollenbesetzungen können Modellunterschiede erklären; daraus folgt kein isolierter Trainingseffekt.", "",
              "Fehlende Traces werden nicht als fehlerfreie Spiele behandelt. Vorhandene Traces werden nicht auf Vollständigkeit geprüft. Ungültige Aktionen werden aus Beobachtungen und Standardregeln rekonstruiert, nicht aus protokollierten Environment-Ereignissen.", "",
              "Bei wenigen Spielen sind die Schätzungen unsicher. Alle Werte sind deskriptiv; Unsicherheitsintervalle und Signifikanztests werden derzeit nicht ausgegeben. „—“ bedeutet keine auswertbaren Gelegenheiten.", "",
              "Für belastbare Trainingsvergleiche: feste Gegner-/Mitspieler-Pools, ausbalancierte Rollen und Positionen, gleiche Generierungsparameter, vollständige Traces und mehrere unabhängige Trainings-Seeds verwenden. Gepaarte Versuche und Unsicherheit der Baseline-Differenzen benötigen ein entsprechend protokolliertes Versuchsdesign.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", nargs="?", type=Path, default=DEFAULT_RESULTS)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", help="Exact checkpoint ID from results.jsonl")
    selection.add_argument("--all-models", action="store_true", help="Analyze every checkpoint")
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-7B-Instruct", help="Reference checkpoint for descriptive winrate differences")
    parser.add_argument("--turns-path", type=Path)
    parser.add_argument("--output", type=Path, help="Save Markdown (.md) or JSON (.json)")
    parser.add_argument("--markdown", type=Path, help="Save a compact Markdown report")
    args = parser.parse_args()
    if args.output and args.output.suffix.lower() not in {".md", ".json"}:
        parser.error("--output requires a .md or .json file")
    try:
        models = sorted({p["checkpoint"] for g in _load(args.results_path) for p in g["players"]}) if args.all_models else [args.model]
        if not models:
            raise ValueError("No checkpoints found in results")
        reports = [analyze_model(args.results_path, model, args.turns_path) for model in models]
        markdown = markdown_report(reports, baseline=args.baseline)
        for destination in [args.output, args.markdown]:
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination == args.output and destination.suffix.lower() == ".json":
                content = json.dumps(reports if args.all_models else reports[0], ensure_ascii=False, indent=2) + "\n"
            else:
                content = markdown
            destination.write_text(content, encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(markdown)


if __name__ == "__main__":
    main()
