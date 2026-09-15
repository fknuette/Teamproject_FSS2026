import argparse
import json
import re
from pathlib import Path
from typing import Optional


ROLE_PATTERN = re.compile(
    r"Welcome to Secret Mafia! You are Player (\d+).*?"
    r"Your role: ([^\n]+).*?Team: ([^\n]+)",
    re.DOTALL,
)
PLAYERS_PATTERN = re.compile(r"Players: (Player \d+(?:, Player \d+)*)")
TARGET_PATTERN = re.compile(r"\[(\d+)\]")
KILLED_PATTERN = re.compile(r"\[GAME\] Player (\d+) was killed during the night\.")
DETECTIVE_PATTERN = re.compile(r"\[GAME\] Player (\d+) IS a ([^\.\n]+)")
ELIMINATED_PATTERN = re.compile(r"\[GAME\] Player (\d+) was eliminated by vote\.")


def _clean(text: str) -> str:
    return text.strip() or "(keine Antwort)"


def _role_name(player_id: int, roles: dict[int, tuple[str, str]]) -> str:
    role = roles.get(player_id, ("Rolle unbekannt", ""))[0]
    return f"Spieler {player_id} ({role})"


def _target(response: str) -> Optional[int]:
    match = TARGET_PATTERN.search(response)
    return int(match.group(1)) if match else None


def _is_vote(row: dict) -> bool:
    return "Voting phase" in row.get("observation", "") and bool(re.fullmatch(r"\s*\[\d+\]\s*", row.get("response", "")))


def _latest_match(rows: list[dict], pattern: re.Pattern) -> Optional[re.Match]:
    matches = []
    for row in rows:
        matches.extend(pattern.finditer(row.get("observation", "")))
    return matches[-1] if matches else None


def _game_summary(rows: list[dict], blocks: list[tuple[list[dict], list[dict]]], roles: dict[int, tuple[str, str]]) -> str:
    final_rewards = {}
    for row in rows:
        player_id = row.get("player_id")
        reward = row.get("reward")
        if isinstance(player_id, int) and isinstance(reward, (int, float)):
            final_rewards[player_id] = reward

    winning_teams = {
        "Mafia" if "mafia" in roles[player_id][1].lower() else "Village"
        for player_id, reward in final_rewards.items()
        if reward > 0 and player_id in roles
    }
    if len(winning_teams) == 1:
        return winning_teams.pop()
    return "nicht eindeutig ermittelbar"


def _night_blocks(rows: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    blocks = []
    current_night = []
    current_day = []
    for row in rows:
        phase = _phase(row)
        if phase == "NACHT":
            if current_night and current_day:
                blocks.append((current_night, current_day))
                current_night = []
                current_day = []
            current_night.append(row)
        elif current_night and phase == "TAG":
            current_day.append(row)
    if current_night:
        blocks.append((current_night, current_day))
    return blocks


def _night_result(night_rows: list[dict], day_rows: list[dict]) -> tuple[Optional[int], dict[int, str]]:
    evidence = " ".join(row.get("observation", "") for row in night_rows + day_rows)
    killed_matches = KILLED_PATTERN.findall(evidence)
    killed = int(killed_matches[-1]) if killed_matches else None
    detective_results = {}
    for target, role in DETECTIVE_PATTERN.findall(evidence):
        detective_results[int(target)] = "Mafia" if "Mafia" in role else role.strip()
    return killed, detective_results


def _eliminated_players(rows: list[dict]) -> set[int]:
    eliminated = set()
    for row in rows:
        observation = row.get("observation", "")
        eliminated.update(int(player_id) for player_id in KILLED_PATTERN.findall(observation))
        eliminated.update(int(player_id) for player_id in ELIMINATED_PATTERN.findall(observation))
    return eliminated


def _append_active_statements(
    lines: list[str],
    heading: str,
    active_players: set[int],
    roles: dict[int, tuple[str, str]],
) -> None:
    lines.extend(["", heading, ""])
    if not active_players:
        lines.append("(keine aktiven Spieler)")
        return
    players = " | ".join(_role_name(player_id, roles) for player_id in sorted(active_players))
    lines.extend([f"- {players}", ""])


def _roles_from_trace(rows: list[dict]) -> dict[int, tuple[str, str]]:
    roles = {}
    for row in rows:
        observation = row.get("observation", "")
        players_match = PLAYERS_PATTERN.search(observation)
        if players_match:
            for player in players_match.group(1).split(", "):
                roles.setdefault(int(player.split()[-1]), ("unbekannt", "unbekannt"))
        match = ROLE_PATTERN.search(observation)
        if match:
            player_id, role, team = match.groups()
            roles[int(player_id)] = (role.strip(), team.strip())
    return roles


def _complete_roles(roles: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
    """Fills roles that never produced their own observation in the trace."""
    completed = dict(roles)
    unknown_players = [
        player_id
        for player_id, (role, _) in sorted(completed.items())
        if role.lower() == "unbekannt"
    ]
    known_roles = {role.lower() for role, _ in completed.values()}
    missing_special_roles = [
        ("Doctor", "Village"),
        ("Detective", "Village"),
    ]
    for special_role, team in missing_special_roles:
        if special_role.lower() not in known_roles and unknown_players:
            player_id = unknown_players.pop(0)
            completed[player_id] = (special_role, team)
            known_roles.add(special_role.lower())

    for player_id in unknown_players:
        completed[player_id] = ("Villager", "Village")
    return completed


def _phase(row: dict) -> str:
    observation = row.get("observation", "")
    markers = [
        (max(observation.rfind("[GAME] Night"), observation.rfind("[GAME] Night phase")), "NACHT"),
        (max(observation.rfind("[GAME] Day breaks"), observation.rfind("[GAME] Day ends")), "TAG"),
    ]
    position, phase = max(markers)
    if position >= 0:
        return phase
    if "Night phase" in observation or "Night has fallen" in observation:
        return "NACHT"
    if "Day breaks" in observation or "DAY phase" in observation:
        return "TAG"
    return "SPIEL"


def format_trace_game(game_id: int, rows: list[dict]) -> list[str]:
    roles = _complete_roles(_roles_from_trace(rows))
    blocks = _night_blocks(rows)
    winner = _game_summary(rows, blocks, roles)
    eliminated_players = _eliminated_players(rows)
    lines = [f"# Spiel {game_id}", "", "## Zusammenfassung", ""]
    lines.append(f"**Ausgang:** {winner} gewinnt  ")
    lines.append(f"**Dauer:** {len(blocks)} Nacht-/Tag-Runden")
    lines.extend(["", "**Spieler:**", ""])
    for player_id in sorted(roles):
        role = roles[player_id][0] if roles[player_id][0] != "unbekannt" else "Rolle unbekannt"
        lines.append(f"- Spieler {player_id} ({role})")

    lines.extend(["", "---", "", "## Spielverlauf", ""])
    eliminated_before_night = set()
    for night_number, (night_rows, day_rows) in enumerate(blocks, start=1):
        killed, detective_results = _night_result(night_rows, day_rows)
        active_during_night = set(roles) - eliminated_before_night
        active_after_night = active_during_night - ({killed} if killed is not None else set())
        lines.extend([f"## Nacht {night_number}", "", "| Spieler | Aktion | Wirkung |", "|---|---|---|"])
        for row in night_rows:
            player_id = row.get("player_id")
            response = _clean(row.get("response", ""))
            if isinstance(player_id, int):
                player_role = roles.get(player_id, ("", ""))[0].lower()
                target = _target(response)
                target_text = _role_name(target, roles) if target is not None else "kein klares Ziel"
                if "mafia" in player_role:
                    action = f"will {target_text} töten"
                    if killed is None:
                        effect = "kein Kill in dieser Nacht"
                    elif target == killed:
                        effect = "Ziel wurde getötet"
                    else:
                        effect = f"nicht dieses Ziel; getötet wurde {_role_name(killed, roles)}"
                elif "doctor" in player_role:
                    action = f"schützt {target_text}"
                    if killed is None:
                        effect = "kein Kill; Schutz hatte keinen sichtbaren Einfluss"
                    elif target == killed:
                        effect = "Schutz erfolgreich; Ziel überlebte"
                    else:
                        effect = f"kein Einfluss; getötet wurde {_role_name(killed, roles)}"
                elif "detective" in player_role:
                    action = f"untersucht {target_text}"
                    target_role = roles.get(target, ("Rolle unbekannt", ""))[0] if target is not None else "Rolle unbekannt"
                    effect = f"Zielrolle: {target_role}"
                else:
                    action = "hat keine Nachtaktion"
                    effect = "keine Aktion"
                lines.append(f"| {_role_name(player_id, roles)} | {action} | {effect} |")

        if killed is not None:
            lines.extend(["", f"**Auflösung:** {_role_name(killed, roles)} wurde in dieser Nacht getötet.", ""])
        else:
            lines.extend(["", "**Auflösung:** In dieser Nacht wurde niemand getötet.", ""])

        _append_active_statements(
            lines,
            "### Aktive Spieler nach der Nacht",
            active_after_night,
            roles,
        )

        if day_rows:
            discussion_rows = [row for row in day_rows if not _is_vote(row)]
            vote_rows = [row for row in day_rows if _is_vote(row)]
            lines.extend(["", "## Tag", "", "### Diskussion", ""])
            for row in discussion_rows:
                player_id = row.get("player_id")
                response = _clean(row.get("response", ""))
                if response != "(keine Antwort)":
                    lines.extend([f"**{_role_name(player_id, roles)}:** {response}", ""])

            if vote_rows:
                lines.extend(["", "### Voting", "", "| Spieler | Stimme für |", "|---|---|"])
                vote_counts = {}
                elected = []
                for row in vote_rows:
                    voter = row.get("player_id")
                    target = _target(row.get("response", ""))
                    if isinstance(target, int):
                        vote_counts[target] = vote_counts.get(target, 0) + 1
                    target_text = _role_name(target, roles) if target is not None else "ungültig"
                    lines.append(f"| {_role_name(voter, roles)} | {target_text} |" )
                if vote_counts:
                    highest = max(vote_counts.values())
                    elected = [target for target, count in vote_counts.items() if count == highest]
                    if len(elected) == 1:
                        lines.extend(["", f"**Ergebnis:** {_role_name(elected[0], roles)} wurde mit {highest} Stimme(n) herausgewählt.", ""])
                    else:
                        tied = ", ".join(_role_name(target, roles) for target in elected)
                        lines.extend(["", f"**Ergebnis:** Gleichstand zwischen {tied}; niemand wurde herausgewählt.", ""])
                _append_active_statements(
                    lines,
                    "### Aktive Spieler nach dem Voting",
                    active_during_night
                    - ({killed} if killed is not None else set())
                    - set(elected),
                    roles,
                )

                eliminated_before_night.update({killed} if killed is not None else set())
                if len(elected) == 1:
                    eliminated_before_night.update(elected)
        else:
            eliminated_before_night.update({killed} if killed is not None else set())

    # Any rows before the first recognized night are still shown as ordinary play.
    if not blocks:
        lines.extend(["### Spiel", ""])
        for row in rows:
            lines.extend([f"**{_role_name(row.get('player_id'), roles)}:** {_clean(row.get('response', ''))}", ""])

    lines.extend(["", "---", "", "## Ergebnis", ""])
    for player_id in sorted(roles):
        team = roles[player_id][1]
        team_result = "GEWONNEN" if team == winner else "VERLOREN"
        status = " | ausgeschieden" if player_id in eliminated_players else " | im Spiel"
        lines.append(f"- {_role_name(player_id, roles)}: {team_result}{status}")
    return lines


def format_legacy_game(data: dict) -> list[str]:
    lines = ["=" * 72, "SPIEL", "=" * 72, "", "ROLLEN"]
    for player_id, info in data["game_info"].items():
        lines.append(f"  Spieler {player_id}: {info['role']}")
    lines.extend(["", "SPIELVERLAUF", ""])
    for turn_number, entry in enumerate(data["log"], start=1):
        lines.extend(
            [
                f"Zug {turn_number} | Spieler {entry['player_id']}",
                "-" * 36,
                _clean(entry["action"]),
                "",
            ]
        )
    lines.extend(["ERGEBNIS", ""])
    for player_id, reward in data["rewards"].items():
        result = "GEWONNEN" if reward == 1 else "VERLOREN"
        lines.append(f"  Spieler {player_id}: {result}")
    return lines


def convert_log(filepath: str, output_path: Optional[str] = None) -> Path:
    input_path = Path(filepath)
    with input_path.open(encoding="utf-8") as source:
        if input_path.suffix == ".jsonl":
            games = {}
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    games.setdefault(row["game_id"], []).append(row)
            lines = []
            for game_id, rows in sorted(games.items()):
                if lines:
                    lines.append("")
                lines.extend(format_trace_game(game_id, rows))
        else:
            lines = format_legacy_game(json.load(source))

    target = Path(output_path) if output_path else input_path.with_suffix(".txt")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _trace_lines(input_path: Path, iteration: Optional[int] = None) -> list[str]:
    games = {}
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                games.setdefault(row["game_id"], []).append(row)

    lines = []
    if iteration is not None:
        lines.extend([f"ITERATION {iteration}", "", ""])
    for game_id, rows in sorted(games.items()):
        if len(lines) > 2:
            lines.append("")
        lines.extend(format_trace_game(game_id, rows))
    return lines


def convert_traces(
    traces_dir: Path,
    iteration: Optional[int],
    output_path: Optional[str],
) -> Path:
    if iteration is None:
        input_paths = sorted(traces_dir.glob("iter_*.jsonl"))
        if not input_paths:
            raise FileNotFoundError(f"Keine iter_*.jsonl-Dateien in {traces_dir}")
        lines = []
        for input_path in input_paths:
            current_iteration = int(input_path.stem.split("_")[-1])
            if lines:
                lines.extend(["", "", "", "", ""])
            lines.extend(_trace_lines(input_path, current_iteration))
        default_output = traces_dir / "all_iterations_readable.md"
    else:
        input_path = traces_dir / f"iter_{iteration}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Trace-Datei nicht gefunden: {input_path}")
        lines = _trace_lines(input_path, iteration)
        default_output = traces_dir / f"iter_{iteration}_readable.md"

    target = Path(output_path) if output_path else default_output
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formatiert Mafia-Spiele als lesbare Textdatei.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Alle Iterationen verarbeiten")
    mode.add_argument("--iteration", type=int, metavar="N", help="Nur iter_N.jsonl verarbeiten")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "scripts" / "runs" / "online_grpo" / "traces",
        help="Ordner mit den iter_*.jsonl-Dateien",
    )
    parser.add_argument("-o", "--output", help="Zieldatei; standardmaessig im Trace-Ordner")
    args = parser.parse_args()
    try:
        output = convert_traces(args.traces_dir, None if args.all else args.iteration, args.output)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(f"Gespeichert: {output}")