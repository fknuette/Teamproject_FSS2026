"""
HumanAgent version: play a FULL game by hand and harvest EVERY voting-phase
observation (across all voting rounds) as .txt scenarios.

This is the mechanics test for the harvesting step: because extraction depends
only on the env (get_observation + phase), not on who plays, HumanAgents let you
verify that phase detection and voting-observation capture work -- no model, no
GPU, and you control the game by hand.

Harvested observations go to a SEPARATE folder (tmp_observation_vote by default)
so they never mix with the fixed observations_vote scenarios. The folder is
emptied first by default -- exactly how the later loop uses it: wipe, then
harvest fresh observations each iteration. Pass --no-clear to keep old files.

Once this works, swap HumanAgent for VLLMTextArenaAgent and the same extraction
logic runs inside the real self-play loop.

Usage:
    python extract_voting_observations_human.py
    python extract_voting_observations_human.py --num-players 8 --seed 45
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# scenarios/ is one level under the project root; src/ holds textarena_utils.
ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import textarena as ta

# Fallback phase detection from the observation text -- guaranteed compatible.
from teamproject_fss2026.textarena_utils import extract_phase

HERE = Path(__file__).resolve().parent


def get_phase(env, observation: str) -> str:
    """
    Determine the current phase robustly: try the direct env attributes first,
    fall back to parsing the observation with your own extract_phase.
    """
    try:
        return env.state.game_state["phase"]
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return env.phase
    except AttributeError:
        pass
    return extract_phase(observation)


def is_voting(phase_value: str) -> bool:
    """Match 'Voting', 'vote', etc."""
    return "vot" in str(phase_value).lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest ALL voting observations from a full game (played by hand)")
    p.add_argument("--env-id", type=str, default="SecretMafia-v0")
    p.add_argument("--num-players", type=int, default=8)
    p.add_argument("--seed", type=int, default=None,
                   help="Optional seed. Default: random (a different game each run).")
    p.add_argument("--output-dir", type=str, default=str(HERE / "tmp_observation_vote"),
                   help="Where harvested voting observations go (kept SEPARATE from the "
                        "fixed observations_vote scenarios).")
    p.add_argument("--no-clear", action="store_true",
                   help="Keep existing files. By default the output dir is emptied first "
                        "so each run/iteration uses only fresh data.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Empty the output dir by default so each run/iteration uses only fresh data.
    # (--no-clear keeps existing files.)
    if not args.no_clear:
        removed = 0
        for old in out_dir.glob("*.txt"):
            old.unlink()
            removed += 1
        print(f"[Clear] removed {removed} old .txt from {out_dir}")

    # All seats are human -> you drive the game to the voting phase yourself.
    agents = {pid: ta.agents.HumanAgent() for pid in range(args.num_players)}

    env = ta.make(env_id=args.env_id)
    try:
        env.reset(num_players=args.num_players, seed=args.seed)
    except TypeError:
        env.reset(num_players=args.num_players)

    print("=" * 70)
    print(f"Env: {args.env_id} | players: {args.num_players} (all human)")
    print("Play the FULL game by hand; every voting-phase observation is captured.")
    print("=" * 70)

    voting_observations: list[dict] = []
    done = False
    turn_id = 0

    while not done:
        player_id, observation = env.get_observation()
        phase = get_phase(env, observation)

        # Always show what's going on so you can see the phase live.
        print("\n" + "-" * 70)
        print(f"[Turn {turn_id}] player {player_id} | phase = {phase}")
        print("-" * 70)
        print(observation)
        print("-" * 70)

        # Capture EVERY voting-phase turn across the whole game. Uniqueness is by
        # turn_id (each turn is unique), NOT by player_id -- the same player votes
        # in multiple rounds, and we want all of those.
        if is_voting(phase):
            voting_observations.append({
                "turn_id": turn_id,
                "player_id": player_id,
                "observation": observation,
            })
            print(f"[Voting] >>> captured observation for player {player_id} (turn {turn_id}) <<<")

        action = agents[player_id](observation)
        done, _ = env.step(action=action)
        turn_id += 1

    # Save each captured voting observation as its own .txt scenario.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = []
    for entry in voting_observations:
        fname = out_dir / f"vote_{stamp}_p{entry['player_id']}_t{entry['turn_id']}.txt"
        fname.write_text(entry["observation"], encoding="utf-8")
        written.append(fname)

    print("\n" + "=" * 70)
    print(f"[Saved] {len(written)} voting observations -> {out_dir}")
    for w in written:
        print(f"  {w.name}")
    print("=" * 70)

    if not voting_observations:
        print("[WARN] No voting observations captured. Either the voting phase "
              "wasn't reached, or is_voting() doesn't match your version's phase "
              "name -- check the 'phase =' values printed above.")


if __name__ == "__main__":
    main()