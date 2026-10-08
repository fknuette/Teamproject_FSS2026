"""
Automatic, role-aware reward assignment for a VOTING scenario.

Rewards are decided HIERARCHICALLY: the checks run top to bottom and the FIRST
matching rule wins (so e.g. a self-vote is always -1, no matter what else holds).

Order of precedence:
  1. invalid vote (no parseable [N])                          -> INVALID   (-1)
  2. self-vote (voted own number)                             -> SELF      (-1)
  3. Mafia voter, target is a teammate (another mafia)        -> TEAMMATE  (-0.5)
  4. Mafia voter, target == a teammate's already-cast vote    -> COORD     (+2)
     (and target is NOT mafia -> real coordination vs village)
  5. Detective voter, target is a CONFIRMED mafia             -> DET_MAFIA (+2)
  6. Detective voter, target is a CONFIRMED innocent          -> DET_INNO  (-1)
  7. Detective voter, target UNKNOWN, but a confirmed mafia is
     still selectable (Valid list) -> missed opportunity      -> DET_MISS  (0)
  8. Mafia voter, target is a non-teammate (village)          -> VILLAGE   (+1)
  9. anything else (e.g. plain villager, non-self valid vote) -> OTHER     (+1)

All info is parsed from the observation the agent saw, so no environment access
is needed. Everything reads the LAST voting phase in the observation (games may
contain several).

Usage:
    from assign_vote_rewards import assign_vote_rewards
    records = assign_vote_rewards("completions/scenario_0.jsonl")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# --- basic parsers (already used before) ----------------------------------- #
_VOTE_NUM_RE = re.compile(r"\[(\d+)\]")                       # a vote like "[3]" -> 3
_SELF_ID_RE = re.compile(r"You are Player (\d+)", re.IGNORECASE)
_ROLE_RE = re.compile(r"Your role:\s*(\w+)", re.IGNORECASE)   # Villager / Mafia / Detective

# --- role-specific parsers ------------------------------------------------- #
# Mafia: "Your teammates are: Player 2, Player 3."
_TEAMMATES_RE = re.compile(r"Your teammates are:\s*([^\n\.]+)")
_PLAYER_NUM_RE = re.compile(r"Player (\d+)")

# Detective: "[GAME] Player 1 IS a Mafia member." / "... IS NOT a Mafia member."
_DET_MAFIA_RE = re.compile(r"Player (\d+) IS a Mafia member", re.IGNORECASE)
_DET_INNOCENT_RE = re.compile(r"Player (\d+) IS NOT a Mafia member", re.IGNORECASE)

# The DAY voting line (not the night "Valid targets"): grab its Valid list.
_VOTING_VALID_RE = re.compile(
    r"Voting phase[^\n]*Valid:\s*([^\n]+)", re.IGNORECASE
)


def extract_vote_target(response: str) -> int | None:
    """Return the voted player number, or None if no valid bracketed vote."""
    m = _VOTE_NUM_RE.search(response)
    return int(m.group(1)) if m else None


def extract_self_id(observation: str) -> int | None:
    m = _SELF_ID_RE.search(observation)
    return int(m.group(1)) if m else None


def extract_role(observation: str) -> str | None:
    m = _ROLE_RE.search(observation)
    return m.group(1).lower() if m else None


def extract_teammates(observation: str, self_id: int) -> set[int]:
    """Mafia teammates OTHER than the voter (the list includes the voter itself)."""
    m = _TEAMMATES_RE.search(observation)
    if not m:
        return set()
    nums = {int(n) for n in _PLAYER_NUM_RE.findall(m.group(1))}
    nums.discard(self_id)   # the observation lists the voter among its teammates
    return nums


def extract_detective_knowledge(observation: str) -> tuple[set[int], set[int]]:
    """Return (confirmed_mafia, confirmed_innocent) from the detective's results.
    Note: check INNOCENT first, since 'IS a Mafia member' is a substring of the
    'IS NOT a Mafia member' line."""
    innocent = {int(n) for n in _DET_INNOCENT_RE.findall(observation)}
    mafia = {int(n) for n in _DET_MAFIA_RE.findall(observation)}
    mafia -= innocent   # safety: never let an innocent leak into the mafia set
    return mafia, innocent


def _last_voting_block(observation: str) -> str:
    """The text from the LAST 'Voting phase' line to the end -- the current round."""
    idxs = [m.start() for m in re.finditer(r"Voting phase", observation, re.IGNORECASE)]
    return observation[idxs[-1]:] if idxs else ""


def extract_valid_targets(observation: str) -> set[int]:
    """Selectable targets in the CURRENT day vote (Valid list of the LAST voting
    phase -- earlier rounds list already-eliminated players as valid)."""
    block = _last_voting_block(observation)
    m = _VOTING_VALID_RE.search(block)
    if not m:
        return set()
    return {int(n) for n in re.findall(r"\[(\d+)\]", m.group(1))}


def extract_prior_votes(observation: str) -> dict[int, int]:
    """Votes already cast in the CURRENT voting round: {voter_id: target_id}.
    Only lines AFTER the last 'Voting phase' marker, of the form '[Player X] [Y]'."""
    block = _last_voting_block(observation)
    votes: dict[int, int] = {}
    for voter, target in re.findall(r"\[Player (\d+)\]\s*\[(\d+)\]", block):
        votes[int(voter)] = int(target)
    return votes


def reward_for_vote(
    observation: str,
    response: str,
    *,
    r_invalid: float = -1.0,
    r_self: float = -1.0,
    r_teammate: float = -0.5,
    r_coord: float = 2.0,
    r_det_mafia: float = 2.0,
    r_det_innocent: float = -1.0,
    r_det_miss: float = 0.0,
    r_village: float = 1.0,
    r_other: float = 1.0,
) -> float:
    """Hierarchical reward: first matching rule wins."""
    self_id = extract_self_id(observation)
    if self_id is None:
        raise ValueError("Could not parse own player number ('You are Player N').")

    target = extract_vote_target(response)

    # 1. invalid
    if target is None:
        return r_invalid
    # 2. self-vote (always wins over everything below)
    if target == self_id:
        return r_self

    role = extract_role(observation)

    # ---- Mafia rules (3, 4, 8) ----
    if role == "mafia":
        teammates = extract_teammates(observation, self_id)
        # 3. voting a teammate is always bad
        if target in teammates:
            return r_teammate
        # 4. coordination: target matches what a teammate already voted
        #    (target is not a teammate here, so it's a real village target)
        prior = extract_prior_votes(observation)
        teammate_targets = {prior[t] for t in teammates if t in prior}
        if target in teammate_targets:
            return r_coord
        # 8. otherwise: mafia voting a non-teammate (village)
        return r_village

    # ---- Detective rules (5, 6, 7) ----
    if role == "detective":
        confirmed_mafia, confirmed_innocent = extract_detective_knowledge(observation)
        # 5. voting a confirmed mafia = optimal
        if target in confirmed_mafia:
            return r_det_mafia
        # 6. voting a confirmed innocent = clear mistake
        if target in confirmed_innocent:
            return r_det_innocent
        # 7. target unknown: missed opportunity if a confirmed mafia is still
        #    selectable in THIS vote; otherwise a fine vote.
        valid = extract_valid_targets(observation)
        selectable_known_mafia = confirmed_mafia & valid
        if selectable_known_mafia:
            return r_det_miss
        return r_other

    # 9. everyone else (plain villager, etc.): valid non-self vote
    return r_other


def assign_vote_rewards(
    input_path: str,
    output_path: str = "",
    **reward_kwargs,
) -> list[dict]:
    """Read a completions JSONL, fill each `reward` hierarchically, write it back
    (in place if output_path is empty), and return the updated records.

    Any of the r_* reward values can be overridden via keyword args, e.g.
    assign_vote_rewards(path, r_coord=3.0)."""
    in_path = Path(input_path)
    records = [
        json.loads(line)
        for line in in_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records found in {in_path}")

    for rec in records:
        rec["reward"] = reward_for_vote(
            rec.get("observation", ""),
            rec.get("response", ""),
            **reward_kwargs,
        )

    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records