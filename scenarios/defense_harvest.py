"""Select accused discussion turns and expose only public game dialogue to judges."""

from __future__ import annotations

import re

PLAYER_LINE_RE = re.compile(r"^\[Player\s+(\d+)\]\s*(.*)$", re.IGNORECASE)
ACCUSATION_RE = re.compile(
    r"\b(?:mafia|verd[aä]chtig|verdacht|auff[aä]llig|schuldig|"
    r"misstrau\w*|ausgewich\w*|widerspruch\w*|erkl[aä]r\w*|"
    r"suspicious|suspect|guilty|evasive|inconsistent|explain|"
    r"vote|voting|lynch|abstimm\w*|stimm\w*|wahl)\b",
    re.IGNORECASE,
)


def public_messages(observation: str) -> list[tuple[int, str]]:
    """Keep only tagged player statements; discard private [GAME] role text."""
    messages = []
    for line in observation.splitlines():
        match = PLAYER_LINE_RE.match(line.strip())
        if match:
            messages.append((int(match.group(1)), match.group(2).strip()))
    return messages


def is_defense_situation(observation: str, player_id: int, window: int = 8) -> bool:
    """Require a recent accusation naming the current player by ID."""
    if window < 1:
        raise ValueError("window must be at least 1")
    target = re.compile(rf"\b(?:(?:Player|Spieler)\s*)?{player_id}\b", re.IGNORECASE)
    for speaker, statement in public_messages(observation)[-window:]:
        if speaker != player_id and target.search(statement) and ACCUSATION_RE.search(statement):
            return True
    return False


def judge_context(observation: str, player_id: int) -> str:
    """Give the judge the target and public dialogue without the player's role."""
    messages = public_messages(observation)
    return f"Beurteile Spieler {player_id}. Öffentliche Aussagen:\n" + "\n".join(
        f"[Player {speaker}] {statement}" for speaker, statement in messages
    )
