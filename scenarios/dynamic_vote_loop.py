"""
Dynamic online GRPO loop for voting.

Each iteration:
  1. HARVEST: the current model plays X full self-play games; from each game up to
     Y voting-phase observations are randomly sampled and frozen as scenarios
     (into a wiped tmp folder -> fresh each iteration). More games = more genuine
     diversity; Y caps how much any single game contributes.
  2. COMPLETIONS: for each harvested scenario, sample N completions and
     auto-assign the self-vote rewards (reused pipeline).
  3. TRAIN: GRPO-train on this iteration's fresh data only (run_training).
  4. MERGE: the merged model becomes the generator for the next iteration.

This is the full dynamic version of vote_generation_loop: same downstream
pipeline, but scenarios are harvested from live self-play instead of read from a
fixed folder. Because every iteration harvests + trains fresh, old_policy == the
model that generated the data stays correct.

Usage:
    python dynamic_vote_loop.py --base-model Qwen/Qwen2.5-7B-Instruct \
        --loop-count 4 --games-per-iter 5 --situations-per-game 4 --bf16
"""

from __future__ import annotations

import argparse
import gc
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import textarena as ta
from vllm import LLM
from transformers import AutoTokenizer

# scenarios/ is one level under the project root.
# grpo_training lives under scripts/, textarena_utils + self_play under src//scripts.
ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from grpo_training.cli import run_training
from completion_generator import generate_completions
from assign_vote_rewards import assign_vote_rewards
from self_play_textarena import VLLMTextArenaAgent
from teamproject_fss2026.textarena_utils import extract_phase

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# phase helpers (same robust detection as the standalone harvest script)
# --------------------------------------------------------------------------- #
def get_phase(env, observation: str) -> str:
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
    return "vot" in str(phase_value).lower()


# --------------------------------------------------------------------------- #
# step 1: harvest voting observations by playing X full games with `model`,
# randomly sampling up to Y voting situations per game.
# reuses an ALREADY-LOADED llm/tokenizer so we don't load the model twice.
# --------------------------------------------------------------------------- #
def harvest_voting_observations(
    llm, tokenizer, env_id: str, num_players: int, out_dir: Path,
    games_per_iter: int = 5, situations_per_game: int = 4,
) -> list[Path]:
    # wipe the tmp dir -> each iteration uses only fresh observations
    removed = 0
    for old in out_dir.glob("*.txt"):
        old.unlink()
        removed += 1
    if removed:
        print(f"[Harvest] cleared {removed} old .txt from {out_dir}")

    agents = {pid: VLLMTextArenaAgent(llm, tokenizer) for pid in range(num_players)}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: list[Path] = []

    for game_idx in range(games_per_iter):
        env = ta.make(env_id=env_id)
        env.reset(num_players=num_players)

        # collect ALL voting observations of this one game first ...
        captured: list[dict] = []
        done = False
        turn_id = 0
        while not done:
            player_id, observation = env.get_observation()
            phase = get_phase(env, observation)
            if is_voting(phase):
                captured.append({"turn_id": turn_id, "player_id": player_id,
                                 "observation": observation})
            agent_out = agents[player_id](observation)
            done, _ = env.step(action=agent_out["action"])
            turn_id += 1

        # ... then randomly sample up to Y of them (fewer if the game had fewer).
        # Random (not the first Y) so we don't systematically favor early rounds.
        if len(captured) > situations_per_game:
            sampled = random.sample(captured, situations_per_game)
        else:
            sampled = captured

        for e in sampled:
            f = out_dir / f"vote_{stamp}_g{game_idx}_p{e['player_id']}_t{e['turn_id']}.txt"
            f.write_text(e["observation"], encoding="utf-8")
            written.append(f)

        print(f"[Harvest] game {game_idx + 1}/{games_per_iter}: "
              f"{len(captured)} voting situations, kept {len(sampled)}")

    print(f"[Harvest] total kept: {len(written)} voting observations "
          f"from {games_per_iter} games")
    return written


def concat_jsonl(files: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for f in files:
            out.write(f.read_text(encoding="utf-8"))


def negative_reward_rate(files: list[Path]) -> tuple[int, int]:
    """Count completions with negative reward (self-vote or invalid) from the JSONLs."""
    import json
    n_neg = n_total = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_total += 1
            if json.loads(line).get("reward", 0.0) < 0:
                n_neg += 1
    return n_neg, n_total


def harvest_and_complete(model: str, tmp_obs_dir: Path, completions_dir: Path,
                         tag: str, args) -> list[Path]:
    """
    Load `model` once, harvest voting observations from a full game, generate
    completions + rewards for each, then free the model. Returns the completion
    JSONL paths. Used both inside the loop and for the final measurement pass.
    `tag` prefixes the completion files (e.g. "iter_1" or "final").
    """
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    scenarios = harvest_voting_observations(
        llm, tokenizer, args.env_id, args.num_players, tmp_obs_dir,
        games_per_iter=args.games_per_iter,
        situations_per_game=args.situations_per_game,
    )
    if not scenarios:
        del llm, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return []

    files: list[Path] = []
    for game_id, scenario in enumerate(scenarios):
        out_file = completions_dir / f"{tag}_{scenario.stem}.jsonl"
        generate_completions(
            scenario=str(scenario),
            model=model,
            player_id=0,
            num_completions=args.num_completions,
            game_id=game_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output=str(out_file),
            llm=llm,
            tokenizer=tokenizer,
        )
        assign_vote_rewards(str(out_file))
        files.append(out_file)

    del llm, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return files


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic online GRPO loop over harvested voting scenarios")
    p.add_argument("--env-id", type=str, default="SecretMafia-v0")
    p.add_argument("--num-players", type=int, default=8)
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--work-dir", type=str, default=str(HERE / "runs" / "dynamic_vote_loop"))
    p.add_argument("--tmp-obs-dir", type=str, default=str(HERE / "tmp_observation_vote"))
    p.add_argument("--completions-dir", type=str, default=str(HERE / "completions_vote_dynamic"))
    p.add_argument("--loop-count", type=int, default=4)
    p.add_argument("--games-per-iter", type=int, default=5,
                   help="X: how many full games to play (harvest from) per iteration. "
                        "More = more genuine scenario diversity, but longer harvest time.")
    p.add_argument("--situations-per-game", type=int, default=4,
                   help="Y: max voting situations randomly sampled from each game. "
                        "Keeps one game from dominating the dataset and caps completion cost.")
    p.add_argument("--num-completions", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    # training hyperparams -> run_training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    ckpt_dir = work_dir / "checkpoints"
    tmp_obs_dir = Path(args.tmp_obs_dir)
    completions_dir = Path(args.completions_dir)
    for d in (data_dir, ckpt_dir, tmp_obs_dir, completions_dir):
        d.mkdir(parents=True, exist_ok=True)

    policy_model = args.base_model
    rate_history: list[tuple[str, int, int]] = []

    for iter_idx in range(1, args.loop_count + 1):
        print(f"\n{'='*70}\n[Iter {iter_idx}] using {policy_model}\n{'='*70}")

        # --- 1+2. HARVEST + COMPLETIONS + REWARDS (model loaded once inside) ---
        iter_files = harvest_and_complete(
            model=policy_model,
            tmp_obs_dir=tmp_obs_dir,
            completions_dir=completions_dir,
            tag=f"iter_{iter_idx}",
            args=args,
        )
        if not iter_files:
            print("[WARN] no voting observations harvested this iteration -- "
                  "the model may be failing to reach the voting phase. Stopping.")
            break

        # negative-vote rate for this iteration (state going INTO training)
        n_neg, n_total = negative_reward_rate(iter_files)
        rate = n_neg / n_total if n_total else 0.0
        rate_history.append((f"iter_{iter_idx}", n_neg, n_total))
        print(f"[Bad-vote] iter {iter_idx}: {n_neg}/{n_total} = {rate:.1%}")

        # --- 3. TRAIN on this iteration's fresh data only ---
        train_file = data_dir / f"train_iter_{iter_idx}.jsonl"
        concat_jsonl(iter_files, train_file)

        adapter_out = ckpt_dir / f"iter_{iter_idx}" / "lora_adapter"
        merged_out = ckpt_dir / f"iter_{iter_idx}" / "merged_model"
        train_args = argparse.Namespace(
            model=policy_model,
            old_policy_model=policy_model,
            data=str(train_file),
            output_dir=str(adapter_out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            bf16=args.bf16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            merge_for_vllm=True,
            merged_output_dir=str(merged_out),
        )
        print(f"[Iter {iter_idx}] training on {train_file}")
        run_training(train_args)
        gc.collect(); torch.cuda.empty_cache()

        # --- 4. next iteration uses the improved model (= online) ---
        prev_policy_model = policy_model
        policy_model = str(merged_out)

        # rolling cleanup: drop the previous merged model (adapters stay as history).
        # runs AFTER run_training (which loads prev_policy_model 3x); never the base.
        if iter_idx > 1 and Path(prev_policy_model).is_dir():
            shutil.rmtree(prev_policy_model, ignore_errors=True)
            print(f"[Cleanup] removed previous merged model: {prev_policy_model}")

    # --- FINAL PASS: measure the fully trained model. The last iteration trained
    # + merged it, but nothing has harvested/completed with it yet. Harvest one
    # more game and complete (NO training, NO merge) just to get its bad-vote rate.
    # policy_model currently points at the last merged model.
    if rate_history:  # only if the loop actually ran at least one iteration
        print(f"\n{'='*70}\n[Final] measuring trained model {policy_model}\n{'='*70}")
        final_files = harvest_and_complete(
            model=policy_model,
            tmp_obs_dir=tmp_obs_dir,
            completions_dir=completions_dir,
            tag="final",
            args=args,
        )
        if final_files:
            n_neg, n_total = negative_reward_rate(final_files)
            final_rate = n_neg / n_total if n_total else 0.0
            rate_history.append(("final", n_neg, n_total))
            print(f"[Bad-vote] final: {n_neg}/{n_total} = {final_rate:.1%}")
        else:
            print("[WARN] final pass harvested no voting observations -- no final rate.")

    # --- summary ---
    print(f"\n{'='*70}\nNegative-vote rate over the run (self-vote or invalid)\n{'='*70}")
    for label, ns, nt in rate_history:
        r = ns / nt if nt else 0.0
        print(f"  {label:>8}: {ns:3d}/{nt:3d} = {r:.1%}")
    print("=" * 70)
    print("\nDynamic vote loop finished.")


if __name__ == "__main__":
    main()