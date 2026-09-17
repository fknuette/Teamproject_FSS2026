"""
LLM version: play a FULL self-play game with the model and harvest EVERY
voting-phase observation (across all voting rounds) as .txt scenarios.

Same harvesting logic as the HumanAgent test, but all seats are played by
VLLMTextArenaAgent -- i.e. real self-play. This is the harvesting step of the
dynamic loop: the current model plays, and we freeze every voting state it
reaches as a scenario for downstream GRPO completions.

Harvested observations go to a SEPARATE folder (tmp_observation_vote by default),
emptied first by default so each run/iteration uses only fresh data.

Usage:
    python extract_voting_observations_llm.py --model Qwen/Qwen2.5-7B-Instruct
    python extract_voting_observations_llm.py --model /path/to/merged_model --seed 45
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# scenarios/ is one level under the project root.
# grpo_training lives under scripts/, textarena_utils under src/ -> add BOTH.
ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import textarena as ta
from vllm import LLM
from transformers import AutoTokenizer

# Reuse the real self-play agent and the phase detection so the game is played
# and read exactly like in production self_play.
from self_play_textarena import VLLMTextArenaAgent
from teamproject_fss2026.textarena_utils import extract_phase

HERE = Path(__file__).resolve().parent


def get_phase(env, observation: str) -> str:
    """Determine the current phase robustly (falls back to extract_phase)."""
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
    p = argparse.ArgumentParser(description="Harvest ALL voting observations from a full self-play game")
    p.add_argument("--env-id", type=str, default="SecretMafia-v0")
    p.add_argument("--num-players", type=int, default=8)
    p.add_argument("--model", type=str, required=True,
                   help="Model the agents use to play the game (path or HF name).")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional seed. Default: random (a different game each run).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--output-dir", type=str, default=str(HERE / "tmp_observation_vote"),
                   help="Where harvested voting observations go (separate from the fixed ones).")
    p.add_argument("--no-clear", action="store_true",
                   help="Keep existing files. By default the output dir is emptied first.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Empty by default so each run/iteration uses only fresh data.
    if not args.no_clear:
        removed = 0
        for old in out_dir.glob("*.txt"):
            old.unlink()
            removed += 1
        print(f"[Clear] removed {removed} old .txt from {out_dir}")

    # --- load the model once; all seats share it (true self-play) ---
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    agents = {pid: VLLMTextArenaAgent(llm, tokenizer) for pid in range(args.num_players)}

    # --- set up the game ---
    env = ta.make(env_id=args.env_id)
    try:
        if args.seed is not None:
            env.reset(num_players=args.num_players, seed=args.seed)
        else:
            env.reset(num_players=args.num_players)
    except TypeError:
        env.reset(num_players=args.num_players)

    print("=" * 70)
    print(f"Env: {args.env_id} | players: {args.num_players} | model: {args.model}")
    print("Playing a full self-play game; every voting-phase observation is captured.")
    print("=" * 70)

    voting_observations: list[dict] = []
    done = False
    turn_id = 0

    while not done:
        player_id, observation = env.get_observation()
        phase = get_phase(env, observation)

        # Capture EVERY voting-phase turn across the whole game (uniqueness by
        # turn_id -- the same player votes in multiple rounds, we want all).
        if is_voting(phase):
            voting_observations.append({
                "turn_id": turn_id,
                "player_id": player_id,
                "observation": observation,
            })
            print(f"[Voting] captured player {player_id} (turn {turn_id})")
        else:
            print(f"[Turn {turn_id}] player {player_id}, phase={phase}")

        # The VLLM agent returns a dict: {"action": ..., "response": ...}
        agent_out = agents[player_id](observation)
        done, _ = env.step(action=agent_out["action"])
        turn_id += 1

    # --- save each captured voting observation as its own .txt scenario ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = []
    for entry in voting_observations:
        fname = out_dir / f"vote_{stamp}_p{entry['player_id']}_t{entry['turn_id']}.txt"
        fname.write_text(entry["observation"], encoding="utf-8")
        written.append(fname)

    print("=" * 70)
    print(f"[Saved] {len(written)} voting observations -> {out_dir}")
    for w in written:
        print(f"  {w.name}")
    print("=" * 70)

    if not voting_observations:
        print("[WARN] No voting observations captured. Either the game didn't reach "
              "a voting phase (agents may be failing on invalid moves), or is_voting() "
              "doesn't match your version's phase name -- check the phase values above.")


if __name__ == "__main__":
    main()