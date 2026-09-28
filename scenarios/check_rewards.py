"""
Verify the reward logic on hand-recorded voting scenarios.

For each .txt scenario it prints, for EVERY possible vote, the reward that
assign_vote_rewards would give -- plus the parsed context (role, teammates,
detective knowledge, valid targets, prior votes) so you can see WHY.

No model, no GPU: this tests the reward FUNCTION, not the model's behaviour.
You know the situation you recorded, so you know the expected rewards.

Usage:
    python check_rewards.py scenarios/tmp_observation_vote/some_scenario.txt
    python check_rewards.py scenarios/tmp_observation_vote/         # whole folder
"""

from __future__ import annotations

import sys
from pathlib import Path

# make the sibling module importable regardless of where we're called from
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from assign_vote_rewards import (
    reward_for_vote,
    extract_self_id,
    extract_role,
    extract_teammates,
    extract_detective_knowledge,
    extract_valid_targets,
    extract_prior_votes,
)


def check_one(path: Path) -> None:
    obs = path.read_text(encoding="utf-8")

    self_id = extract_self_id(obs)
    role = extract_role(obs)
    valid = extract_valid_targets(obs)
    prior = extract_prior_votes(obs)

    print("=" * 70)
    print(f"Scenario: {path.name}")
    print(f"  self_id      : {self_id}")
    print(f"  role         : {role}")
    print(f"  valid targets: {sorted(valid)}")
    print(f"  prior votes  : {prior}")
    if role == "mafia" and self_id is not None:
        print(f"  teammates    : {sorted(extract_teammates(obs, self_id))}")
    if role == "detective":
        mafia, inno = extract_detective_knowledge(obs)
        print(f"  known mafia  : {sorted(mafia)}")
        print(f"  known innocent: {sorted(inno)}")
    print("-" * 70)

    # try every player number that appears as a valid target (fallback 0..9)
    candidates = sorted(valid) if valid else list(range(10))
    for target in candidates:
        try:
            r = reward_for_vote(obs, f"[{target}]")
            marker = " <- SELF" if target == self_id else ""
            print(f"  vote [{target}] -> reward {r:+.1f}{marker}")
        except Exception as e:
            print(f"  vote [{target}] -> ERROR: {e}")
    # also test an invalid (unparseable) response
    print(f"  (no bracket) -> reward {reward_for_vote(obs, 'I abstain'):+.1f}  [invalid case]")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python check_rewards.py <scenario.txt | folder>")

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.txt"))
        if not files:
            raise SystemExit(f"No .txt files in {target}")
        for f in files:
            check_one(f)
    else:
        check_one(target)


if __name__ == "__main__":
    main()
