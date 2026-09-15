"""
Automatic reward assignment for a VOTING scenario.

Rule:
    - self-vote            -> reward = -1.0  (voted own number)
    - any other valid vote -> reward = +1.0
    - no parseable vote    -> reward = -1.0  (configurable)

The vote is parsed with the game's own parse_model_response, so the reward is
based on the action the game would actually score. The agent's own number is
taken from the observation's opening line ("You are Player N"), not from metadata.

Usage:
    from assign_vote_rewards import assign_vote_rewards
    records = assign_vote_rewards("completions/scenario_0.jsonl")
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teamproject_fss2026.textarena_utils import parse_model_response

_VOTE_NUM_RE = re.compile(r"\[(\d+)\]")            # voted number, e.g. "[3]" -> 3
_SELF_ID_RE = re.compile(r"You are Player (\d+)", re.IGNORECASE)  # own number


def extract_vote_target(response: str) -> int | None:
    """Return the voted player number, or None if no valid bracketed vote."""
    m = _VOTE_NUM_RE.search(parse_model_response(response).action)
    return int(m.group(1)) if m else None


def extract_self_id(observation: str) -> int | None:
    """Return the agent's own player number from the observation, or None."""
    m = _SELF_ID_RE.search(observation)
    return int(m.group(1)) if m else None


def assign_vote_rewards(
    input_path: str,
    output_path: str = "",
    self_vote_reward: float = -1.0,
    other_reward: float = 1.0,
    invalid_reward: float = -1.0,
) -> list[dict]:
    """
    Read a completions JSONL, fill each `reward` per the voting rule, write it
    back out (in place if output_path is empty), and return the updated records.
    """
    in_path = Path(input_path)
    records = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {in_path}")

    for rec in records:
        voter_id = extract_self_id(rec.get("observation", ""))
        if voter_id is None:
            # Fail loudly: silently skipping the self-vote check would wrongly
            # reward real self-votes with +1.
            raise ValueError(
                "Could not parse own player number ('You are Player N') for "
                f"game_id={rec.get('game_id')}, turn_id={rec.get('turn_id')}."
            )

        target = extract_vote_target(rec.get("response", ""))
        if target is None:
            rec["reward"] = invalid_reward
        elif target == voter_id:
            rec["reward"] = self_vote_reward
        else:
            rec["reward"] = other_reward

    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records