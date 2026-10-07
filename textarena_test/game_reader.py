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
# Match SecretMafiaEnv.voting_pattern exactly, including its greedy selection.
TARGET_PATTERN = re.compile(r".*\[(?:player\s*)?(\d+)\].*", re.IGNORECASE)
# Enumerate every option in the observation, without the action parser's greediness.
VALID_TARGET_PATTERN = re.compile(r"\[(\d+)\]")
VALID_VOTES_PATTERN = re.compile(r"Valid:\s*([^\n]+)")
KILLED_PATTERN = re.compile(r"\[GAME\] Player (\d+) was killed during the night\.")
DETECTIVE_PATTERN = re.compile(r"\[GAME\] Player (\d+) IS a ([^\.\n]+)")
INVALID_ELIMINATED_PATTERN = re.compile(r"\[GAME\] Player (\d+) has been eliminated by making an invalid move\.")
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
    return _phase(row) == "VOTING"


def _vote_target(row: dict) -> Optional[int]:
    target = _target(row.get("response", ""))
    # Observations include earlier rounds; validate against the latest options.
    valid_votes = list(VALID_VOTES_PATTERN.finditer(row.get("observation", "")))
    if valid_votes and target not in {int(player_id) for player_id in VALID_TARGET_PATTERN.findall(valid_votes[-1].group(1))}:
        return None
    return target


def _action_outcomes(rows: list[dict], roles: Optional[dict[int, tuple[str, str]]] = None) -> dict[int, dict]:
    """Replay target validity and retries using Secret Mafia's default rules.

    Observations precede actions and contain repeated history. Count each action
    once, not each warning. Infer a fatal second error even on the final row,
    whose resulting observation is absent from the trace.
    """
    roles = _roles_from_trace(rows) if roles is None else roles
    alive = set(roles)
    outcomes = {}
    previous_invalid_player = None
    for row in rows:
        alive -= _eliminated_players([row])
        player = row.get("player_id")
        phase = _phase(row)
        role = roles.get(player, ("", ""))[0].lower()
        targeted = phase == "VOTING" or (
            phase == "NACHT" and role in {"mafia", "doctor", "detective"}
        )
        target = _target(row.get("response", ""))
        invalid = targeted and (target is None or target not in alive)
        eliminated = invalid and previous_invalid_player == player
        reason = "kein gültiges Zielformat" if target is None else "Ziel ist nicht im Spiel"
        outcomes[id(row)] = {
            "target": target, "invalid": invalid, "eliminated": eliminated,
            "reason": reason if invalid else "",
        }
        if eliminated:
            alive.discard(player)
        previous_invalid_player = player if invalid and not eliminated else None
    return outcomes


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
        elif current_night and phase in {"TAG", "VOTING"}:
            current_day.append(row)
    if current_night:
        blocks.append((current_night, current_day))
    return blocks


def _night_result(night_rows: list[dict], day_rows: list[dict]) -> tuple[Optional[int], dict[int, str]]:
    killed = None
    for row in night_rows + day_rows:
        observation = row.get("observation", "")
        night_start = max(observation.rfind("[GAME] Night"), observation.rfind("[GAME] Night phase"))
        if night_start < 0:
            continue
        current_kills = [
            int(match.group(1))
            for match in KILLED_PATTERN.finditer(observation)
            if match.start() > night_start
        ]
        if current_kills:
            killed = current_kills[-1]

    evidence = " ".join(row.get("observation", "") for row in night_rows + day_rows)
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
        eliminated.update(int(player_id) for player_id in INVALID_ELIMINATED_PATTERN.findall(observation))
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


def _discussion_rounds(rows: list[dict], active_players: set[int]) -> list[list[dict]]:
    rounds = []
    current_round = []
    spoken_players = set()
    for row in rows:
        player_id = row.get("player_id")
        if current_round and isinstance(player_id, int) and player_id in spoken_players:
            rounds.append(current_round)
            current_round = []
            spoken_players = set()
        current_round.append(row)
        if isinstance(player_id, int) and player_id in active_players:
            spoken_players.add(player_id)
        if active_players and spoken_players == active_players:
            rounds.append(current_round)
            current_round = []
            spoken_players = set()
    if current_round:
        rounds.append(current_round)
    return rounds


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
        (observation.rfind("[GAME] Voting phase"), "VOTING"),
    ]
    position, phase = max(markers)
    if position >= 0:
        return phase
    if "Night phase" in observation or "Night has fallen" in observation:
        return "NACHT"
    if "Day breaks" in observation or "DAY phase" in observation:
        return "TAG"
    if "Voting phase" in observation:
        return "VOTING"
    return "SPIEL"


def format_trace_game(game_id: int, rows: list[dict]) -> list[str]:
    roles = _complete_roles(_roles_from_trace(rows))
    blocks = _night_blocks(rows)
    winner = _game_summary(rows, blocks, roles)
    outcomes = _action_outcomes(rows)
    invalid_players = {row.get("player_id") for row in rows if outcomes[id(row)]["eliminated"]}
    eliminated_players = _eliminated_players(rows) | invalid_players
    lines = ["# Welcome to Secret Mafia", "", f"## Spiel {game_id}", "", "## Zusammenfassung", ""]
    lines.append(f"**Ausgang:** {winner} gewinnt  ")
    lines.append(f"**Dauer:** {len(blocks)} Nacht-/Tag-Runden")
    lines.extend(["", "**Spieler:**", ""])
    for player_id in sorted(roles):
        role = roles[player_id][0] if roles[player_id][0] != "unbekannt" else "Rolle unbekannt"
        lines.append(f"- Spieler {player_id} ({role})")

    lines.extend(["", "---", "", "## Spielverlauf", ""])
    active_players = set(roles)
    for night_number, (night_rows, day_rows) in enumerate(blocks, start=1):
        killed, detective_results = _night_result(night_rows, day_rows)
        active_before_night = active_players.copy()
        night_invalid = {row.get("player_id") for row in night_rows if outcomes[id(row)]["eliminated"]}
        if killed not in active_before_night:
            killed = None
        mafia_targets = {
            target
            for row in night_rows
            if "mafia" in roles.get(row.get("player_id"), ("", ""))[0].lower()
            for target in [outcomes[id(row)]["target"]]
            if target is not None and not outcomes[id(row)]["invalid"]
        }
        doctor_targets = {
            target
            for row in night_rows
            if "doctor" in roles.get(row.get("player_id"), ("", ""))[0].lower()
            for target in [outcomes[id(row)]["target"]]
            if target is not None and not outcomes[id(row)]["invalid"]
        }
        # A matching doctor and Mafia target means the night ended with no kill.
        # This takes precedence over stale kill messages in later observations.
        if mafia_targets & doctor_targets:
            killed = None
        candidate_active_after_night = active_before_night - ({killed} if killed is not None else set())
        if candidate_active_after_night == active_before_night:
            killed = None
        active_after_night = active_before_night - ({killed} if killed is not None else set()) - night_invalid
        active_players = active_after_night.copy()
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
                        effect = "Ziel wurde geschützt" if target in doctor_targets else "kein Kill in dieser Nacht"
                    elif target == killed:
                        effect = "Ziel wurde getötet"
                    else:
                        effect = f"nicht dieses Ziel; getötet wurde {_role_name(killed, roles)}"
                elif "doctor" in player_role:
                    action = f"schützt {target_text}"
                    if killed is None:
                        effect = (
                            "Schutz erfolgreich; Ziel überlebte"
                            if target in mafia_targets
                            else "kein Kill; Schutz hatte keinen sichtbaren Einfluss"
                        )
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
                outcome = outcomes[id(row)]
                if outcome["invalid"]:
                    action = f"ungültige Aktion: {action} ({outcome['reason']})"
                    effect = (
                        "wegen zweiter ungültiger Eingabe ausgeschieden"
                        if outcome["eliminated"] else "nicht ausgeführt; erneuter Versuch erlaubt"
                    )
                lines.append(f"| {_role_name(player_id, roles)} | {action} | {effect} |")

        if killed is not None:
            lines.extend(["", f"**Auflösung:** {_role_name(killed, roles)} wurde in dieser Nacht getötet.", ""])
        else:
            lines.extend(["", "**Auflösung:** Kein Mafia-Kill in dieser Nacht.", ""])
        for player_id in sorted(night_invalid):
            lines.extend([f"**Ausgeschieden:** {_role_name(player_id, roles)} wegen zweiter ungültiger Eingabe.", ""])

        _append_active_statements(
            lines,
            "### Aktive Spieler nach der Nacht",
            active_after_night,
            roles,
        )

        if day_rows:
            discussion_rows = [row for row in day_rows if not _is_vote(row)]
            vote_rows = [row for row in day_rows if _is_vote(row)]
            active_during_day = active_after_night
            lines.extend(["", "## Tag", "", "### Diskussion", ""])
            for round_number, discussion_round in enumerate(
                _discussion_rounds(discussion_rows, active_during_day),
                start=1,
            ):
                lines.extend([f"#### Diskussionsrunde {round_number}", ""])
                for row in discussion_round:
                    player_id = row.get("player_id")
                    response = _clean(row.get("response", ""))
                    if response != "(keine Antwort)":
                        lines.extend([f"**{_role_name(player_id, roles)}:** {response}", ""])

            if vote_rows:
                lines.extend(["", "### Voting", "", "| Spieler | Aktion | Stimme für |", "|---|---|---|"])
                vote_counts = {}
                elected = []
                attempts_by_player = {}
                for row in vote_rows:
                    attempts_by_player.setdefault(row.get("player_id"), []).append(row)
                invalid_eliminated = {
                    row.get("player_id") for row in vote_rows
                    if outcomes[id(row)]["eliminated"]
                }
                for voter, attempts in attempts_by_player.items():
                    actions = []
                    counted_target = None
                    for row in attempts:
                        raw_target = _target(row.get("response", ""))
                        target = None if outcomes[id(row)]["invalid"] else outcomes[id(row)]["target"]
                        if target is None:
                            action = (
                                f"ungültige Stimme für {_role_name(raw_target, roles)}"
                                if raw_target is not None
                                else "ungültige Stimme (kein erkennbares Ziel)"
                            )
                        else:
                            action = f"gültige Stimme für {_role_name(target, roles)}"
                            counted_target = target
                        actions.append(action)
                    if voter in invalid_eliminated:
                        actions.append("wegen ungültiger Eingabe ausgeschieden")
                    if counted_target is not None:
                        vote_counts[counted_target] = vote_counts.get(counted_target, 0) + 1
                    target_text = _role_name(counted_target, roles) if counted_target is not None else "keine gültige Stimme"
                    lines.append(f"| {_role_name(voter, roles)} | {' → danach '.join(actions)} | {target_text} |")
                if vote_counts:
                    highest = max(vote_counts.values())
                    elected = [target for target, count in vote_counts.items() if count == highest]
                    eliminated_by_vote = [
                        int(player_id)
                        for row in day_rows + (blocks[night_number][0] if night_number < len(blocks) else [])
                        for player_id in ELIMINATED_PATTERN.findall(row.get("observation", ""))
                    ]
                    selected_on_tie = eliminated_by_vote[-1] if eliminated_by_vote else None
                    if len(elected) == 1:
                        lines.extend(["", f"**Ergebnis:** {_role_name(elected[0], roles)} wurde mit {highest} Stimme(n) herausgewählt.", ""])
                        voted_out = elected
                    elif selected_on_tie in elected:
                        lines.extend(
                            [
                                "",
                                f"**Ergebnis:** Gleichstand zwischen {', '.join(_role_name(target, roles) for target in elected)}; "
                                f"zufällig wurde {_role_name(selected_on_tie, roles)} herausgewählt.",
                                "",
                            ]
                        )
                        voted_out = [selected_on_tie]
                    else:
                        tied = ", ".join(_role_name(target, roles) for target in elected)
                        lines.extend(["", f"**Ergebnis:** Gleichstand zwischen {tied}; niemand wurde herausgewählt.", ""])
                        voted_out = []
                else:
                    voted_out = []
                _append_active_statements(
                    lines,
                    "### Aktive Spieler nach dem Voting",
                    active_after_night - set(voted_out) - invalid_eliminated,
                    roles,
                )

                active_players = active_after_night - set(voted_out) - invalid_eliminated
        else:
            active_players = active_after_night

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
        if player_id in invalid_players:
            status += " (wegen ungültiger Eingaben)"
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

    target = Path(output_path) if output_path else input_path.with_suffix(".md")
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


def convert_directory(input_dir: Path, output_path: Optional[str] = None) -> Path:
    input_paths = sorted(input_dir.glob("*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"Keine .jsonl-Dateien in {input_dir}")

    lines = []
    for input_path in input_paths:
        if lines:
            lines.extend(["", "", "", "", ""])
        lines.extend([f"DATEI {input_path.name}", "", ""])
        lines.extend(_trace_lines(input_path))

    target = Path(output_path) if output_path else input_dir / f"{input_dir.name}_readable.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def convert_input(input_path: Path, output_path: Optional[str] = None) -> Path:
    if input_path.is_file():
        return convert_log(str(input_path), output_path)
    if input_path.is_dir():
        return convert_directory(input_path, output_path)
    raise FileNotFoundError(f"Eingabepfad nicht gefunden: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formatiert Mafia-Spiele als lesbare Textdatei.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Alle Iterationen verarbeiten")
    mode.add_argument("--iteration", type=int, metavar="N", help="Nur iter_N.jsonl verarbeiten")
    mode.add_argument(
        "--input",
        "--file",
        dest="input",
        type=Path,
        metavar="PFAD",
        help="Eine einzelne Datei oder einen Ordner mit .jsonl-Dateien verarbeiten",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "scripts" / "runs" / "online_grpo" / "traces",
        help="Ordner mit den iter_*.jsonl-Dateien",
    )
    parser.add_argument("-o", "--output", help="Zieldatei; standardmaessig neben der Eingabe")
    args = parser.parse_args()

    try:
        if args.input:
            output = convert_input(args.input, args.output)
        else:
            output = convert_traces(args.traces_dir, None if args.all else args.iteration, args.output)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(f"Gespeichert: {output}")