"""Joint dynamic GRPO loop for vote and defense tasks.

This file combines the dynamic vote loop and the defense loop into a single
training signal. The idea is to keep one optimizer step over a mixed batch that
contains samples from both action types, instead of alternating between separate
training phases.

Weighted combined objective:
    L = lambda_vote * L_vote + lambda_disc * L_disc

where each component computes its own GRPO loss with its own groupwise reward
normalization. This is the stable variant recommended for LoRA fine-tuning when
samples from voting and defense are produced from the same self-play games.

Usage:
    python dynamic_joint_loop.py \
        --vote-data runs/dynamic_vote_loop/data/train_iter_1.jsonl \
        --defense-data runs/defense_loop/data/train_iter_1.jsonl \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --old-policy-model Qwen/Qwen2.5-7B-Instruct \
        --epochs 1 --batch-size 2 --bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import textarena as ta
import torch
from vllm import LLM
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for subdir in ("src", "scripts"):
    module_dir = str(ROOT / subdir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from dynamic_vote_loop import get_phase, is_voting
from dynamic_defense_loop import extract_roles, select_judge_ids
from assign_defense_rewards import assign_team_aware_rewards
from assign_vote_rewards import assign_vote_rewards
from completion_generator import generate_completions
from defense_harvest import judge_context
from defense_judge import evaluate_all_suspicions, make_evaluator
from self_play_textarena import VLLMTextArenaAgent
from teamproject_fss2026.textarena_utils import build_agent_prompt, extract_phase


@dataclass
class JointSample:
    action_type: str  # "vote" or "defense"
    observation: str
    response: str
    reward: float
    advantage: float
    game_id: int
    player_id: int = 0
    turn_id: int = 0


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout file: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rollout data found in {path}")
    return rows


def normalize_group_advantages(records: Iterable[dict]) -> list[dict]:
    """Group-relative normalization within a game / group.

    This matches the repo's GRPO convention: advantages are computed relative to
    the reward distribution inside each group, not globally across all samples.
    """
    records = list(records)
    if not records:
        return []

    by_group: dict[int, list[dict]] = {}
    for row in records:
        group_key = int(row.get("game_id", 0))
        by_group.setdefault(group_key, []).append(row)

    normalized: list[dict] = []
    for group_id, group_rows in by_group.items():
        rewards = [float(r["reward"]) for r in group_rows]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance) if variance > 0 else 1.0
        if std_reward < 1e-8:
            std_reward = 1.0

        for row in group_rows:
            row = dict(row)
            row["advantage"] = (float(row["reward"]) - mean_reward) / std_reward
            normalized.append(row)

    return normalized


def apply_reward_to_records(records: list[dict], action_type: str) -> list[dict]:
    """Apply the existing reward rule for the action type.

    This keeps all reward assignment in the rollout stage rather than splitting
    the logic across compute_vote_reward/compute_defense_reward helpers.
    """
    action_type = str(action_type).lower()
    if action_type == "vote":
        temp_path = Path(tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name)
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored = assign_vote_rewards(str(temp_path))
            return scored
        finally:
            temp_path.unlink(missing_ok=True)

    if action_type == "defense":
        if not records:
            return []
        return assign_team_aware_rewards(records, output_path="")

    return records


def debug_summary(rows: list[dict], label: str) -> None:
    """Print a concise reward/phase summary for mixed-loop debugging."""
    if not rows:
        print(f"[Debug] {label}: 0 rows")
        return

    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        action_type = str(row.get("action_type", "vote")).lower()
        by_action[action_type].append(row)

    print(f"[Debug] {label}: total={len(rows)}")
    for action_type, group in sorted(by_action.items()):
        rewards = [float(r.get("reward", 0.0)) for r in group if "reward" in r]
        if rewards:
            mean_reward = sum(rewards) / len(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)
            print(
                f"[Debug] {label} action={action_type}: count={len(group)} "
                f"reward_mean={mean_reward:.4f} reward_min={min_reward:.4f} reward_max={max_reward:.4f}"
            )
        else:
            print(f"[Debug] {label} action={action_type}: count={len(group)} reward_missing")


def score_joint_rollouts(rows: list[dict]) -> list[dict]:
    """Attach task-specific rewards and keep a unified dataset layout."""
    scored: list[dict] = []
    for row in rows:
        action_type = str(row.get("action_type", "vote")).lower()
        row = dict(row)
        row["action_type"] = action_type
        if action_type == "vote":
            row["reward"] = 0.0
        else:
            row["reward"] = 0.0
        scored.append(row)

    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_action[str(row.get("action_type", "vote")).lower()].append(row)

    final_rows: list[dict] = []
    for action_type, group_rows in by_action.items():
        final_rows.extend(apply_reward_to_records(group_rows, action_type))

    debug_summary(final_rows, "scored_joint_rollouts")
    return final_rows


def _resolve_artifact_paths(row: dict, action_type: str, artifact_root: Path | None = None) -> tuple[Path | None, Path | None]:
    """Return stable artifact paths for the scenario and its completion JSONL."""
    if artifact_root is None:
        return None, None

    artifact_root = Path(artifact_root)
    scenario_dir = artifact_root / "observations"
    completion_dir = artifact_root / "completions"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    completion_dir.mkdir(parents=True, exist_ok=True)

    game_id = int(row.get("game_id", 0))
    player_id = int(row.get("player_id", 0))
    turn_id = int(row.get("turn_id", 0))
    stem = f"{action_type}_g{game_id}_p{player_id}_t{turn_id}"
    scenario_path = scenario_dir / f"{stem}.txt"
    output_path = completion_dir / f"{stem}.jsonl"
    return scenario_path, output_path


def generate_joint_rollouts(model: str, rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Generate completions for harvested vote/defense scenarios and reward them.

    The actor runtime is kept alive only for the generation pass. Once all
    completions are produced, the actor is torn down before judge evaluation so
    the local judge does not overlap the same vLLM model instance on GPU memory.
    """
    artifact_root = Path(args.output_dir) / "artifacts" if getattr(args, "output_dir", None) else None
    judge_gpu_memory_utilization = (
        min(float(args.gpu_memory_utilization), 0.35)
        if getattr(args, "judge_mode", "local") == "local"
        else float(args.gpu_memory_utilization)
    )
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = None
    try:
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model)
    except Exception:
        tokenizer = None

    generated: list[dict] = []
    for idx, row in enumerate(rows):
        action_type = str(row.get("action_type", "vote")).lower()
        scenario = row.get("observation", "")
        if not isinstance(scenario, str) or not scenario.strip():
            continue

        scenario_path, output_path = _resolve_artifact_paths(row, action_type, artifact_root)
        if scenario_path is not None:
            scenario_path.write_text(scenario, encoding="utf-8")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            records = generate_completions(
                scenario=str(scenario_path) if scenario_path is not None else scenario,
                model=model,
                player_id=int(row.get("player_id", 0)),
                num_completions=int(args.num_completions),
                game_id=int(row.get("game_id", idx)),
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                output=str(output_path) if output_path is not None else None,
                llm=llm,
                tokenizer=tokenizer,
            )
            for rec in records:
                rec["action_type"] = action_type
                rec["game_id"] = int(row.get("game_id", rec.get("game_id", idx)))
                rec["player_id"] = int(row.get("player_id", rec.get("player_id", 0)))
                rec["turn_id"] = int(row.get("turn_id", rec.get("turn_id", 0)))
                rec["observation"] = scenario
                rec["judge_observation"] = row.get("judge_observation", scenario)
                rec["player_team"] = row.get("player_team", "Village")
                rec["judge_ids"] = row.get("judge_ids", [])
                rec["all_player_ids"] = row.get("all_player_ids", [])
                rec["villager_ids"] = row.get("villager_ids", [])
                rec["mafia_ids"] = row.get("mafia_ids", [])
                rec["suspicion_pre"] = row.get("suspicion_pre", {})
                generated.append(rec)
        finally:
            if scenario_path is not None and scenario_path.exists():
                pass
            if output_path is not None and output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink(missing_ok=True)

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    by_action: dict[str, list[dict]] = defaultdict(list)
    for rec in generated:
        by_action[str(rec.get("action_type", "vote")).lower()].append(rec)

    if by_action.get("vote"):
        vote_temp = Path(tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name)
        try:
            with vote_temp.open("w", encoding="utf-8") as handle:
                for rec in by_action["vote"]:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            assign_vote_rewards(str(vote_temp), output_path=str(vote_temp))
            scored_votes = [json.loads(line) for line in vote_temp.read_text(encoding="utf-8").splitlines() if line.strip()]
            by_action["vote"] = scored_votes
        finally:
            vote_temp.unlink(missing_ok=True)

    if by_action.get("defense"):
        evaluator = make_evaluator(
            getattr(args, "judge_mode", "local"),
            getattr(args, "judge_model", ""),
            judge_gpu_memory_utilization,
        )
        assign_team_aware_rewards(by_action["defense"], output_path="", evaluator=evaluator)

    combined = []
    for action_type in ("vote", "defense"):
        combined.extend(by_action.get(action_type, []))
    debug_summary(combined, "generated_joint_rollouts")
    return combined


def build_joint_dataset(vote_path: Path | None = None, defense_path: Path | None = None, rows: list[dict] | None = None) -> list[JointSample]:
    """Build one dataset from vote + defense rollouts.

    This supports both static JSONL files and an in-memory mixed batch. The
    resulting samples are normalized per task and then mixed together in the same
    GRPO optimization loop with a 1:1 loss weight.
    """
    if rows is None:
        rows = []
        if vote_path is not None:
            rows.extend(load_jsonl(vote_path))
        if defense_path is not None:
            rows.extend(load_jsonl(defense_path))

    scored_rows = score_joint_rollouts(rows)
    if not scored_rows:
        print("[Debug] build_joint_dataset: no scored rows after reward assignment")
        return []

    debug_summary(scored_rows, "dataset_before_normalization")

    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        if row.get("response") is None or not str(row["response"]).strip():
            continue
        action_type = str(row.get("action_type", "vote")).lower()
        row["action_type"] = action_type
        by_action[action_type].append(row)

    normalized_rows: list[dict] = []
    for action_type, records in by_action.items():
        normalized_rows.extend(normalize_group_advantages(records))

    samples: list[JointSample] = []
    for row in normalized_rows:
        samples.append(
            JointSample(
                action_type=str(row.get("action_type", "vote")),
                observation=str(row.get("observation", "")),
                response=str(row.get("response", "")),
                reward=float(row.get("reward", 0.0)),
                advantage=float(row.get("advantage", 0.0)),
                game_id=int(row.get("game_id", 0)),
                player_id=int(row.get("player_id", 0)),
                turn_id=int(row.get("turn_id", 0)),
            )
        )

    print(f"[Debug] build_joint_dataset: final_samples={len(samples)}")
    if samples:
        reward_summary = {
            "vote": [s.reward for s in samples if s.action_type.lower() == "vote"],
            "defense": [s.reward for s in samples if s.action_type.lower() == "defense"],
        }
        for action_type, rewards in reward_summary.items():
            if rewards:
                print(
                    f"[Debug] final_sample_rewards action={action_type}: "
                    f"count={len(rewards)} mean={sum(rewards)/len(rewards):.4f} min={min(rewards):.4f} max={max(rewards):.4f}"
                )
    return samples




def _annotate_defense_row(
    row: dict,
    env,
    evaluator=None,
    judge_mode: str = "local",
    judge_model: str = "",
    gpu_memory_utilization: float = 0.6,
) -> dict:
    """Attach the metadata that the defense reward path expects.

    The joint loop harvests the observation, but it still needs the same role and
    judge metadata that the standalone defense loop produces before reward
    assignment.
    """
    row = dict(row)
    player_id = int(row.get("player_id", 0))
    role_map = extract_roles(env)
    if not role_map:
        return row

    all_player_ids = sorted(role_map.keys())
    villager_ids = [pid for pid in all_player_ids if role_map[pid]["team"] == "Village"]
    mafia_ids = [pid for pid in all_player_ids if role_map[pid]["team"] == "Mafia"]
    player_team = role_map.get(player_id, {}).get("team", "Village")

    row["player_team"] = player_team
    row["all_player_ids"] = all_player_ids
    row["villager_ids"] = villager_ids
    row["mafia_ids"] = mafia_ids
    row["judge_observation"] = judge_context(str(row.get("observation", "")), player_id)
    row["judge_ids"] = select_judge_ids(player_id, villager_ids, max_judges=3)

    if evaluator is None:
        judge_util = min(float(gpu_memory_utilization), 0.35) if judge_mode == "local" else float(gpu_memory_utilization)
        evaluator = make_evaluator(judge_mode, judge_model, judge_util)
    row["suspicion_pre"] = evaluate_all_suspicions(
        observation=str(row.get("observation", "")),
        public_game_state=str(row.get("judge_observation", "")),
        response=None,
        judge_ids=row["judge_ids"],
        all_player_ids=all_player_ids,
        evaluator=evaluator,
    )
    return row


def harvest_joint_observations(
    env_id: str,
    num_players: int,
    games_per_iter: int,
    situations_per_game: int,
    model_name: str | None = None,
    artifact_root: str | Path | None = None,
    judge_mode: str = "local",
    judge_model: str = "",
    gpu_memory_utilization: float = 0.6,
) -> list[dict]:
    """Harvest both vote and defense scenarios from live self-play games.

    If a model is provided, we use the real VLLM textarena agent so the game
    actually progresses through voting and discussion phases. Without a model,
    we fall back to a lightweight noop pass action for debugging only.
    """
    artifact_root = Path(artifact_root) if artifact_root is not None else None
    if artifact_root is not None:
        (artifact_root / "observations").mkdir(parents=True, exist_ok=True)
        (artifact_root / "completions").mkdir(parents=True, exist_ok=True)

    judge_gpu_memory_utilization = (
        min(float(gpu_memory_utilization), 0.35) if judge_mode == "local" else float(gpu_memory_utilization)
    )
    defense_evaluator = None

    harvested: list[dict] = []
    llm = None
    tokenizer = None
    agents = None

    if model_name is not None:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        agents = {pid: VLLMTextArenaAgent(llm, tokenizer) for pid in range(num_players)}

    selected_vote_rows: list[dict] = []
    selected_defense_rows: list[dict] = []

    try:
        for game_idx in range(games_per_iter):
            env = ta.make(env_id=env_id)
            env.reset(num_players=num_players)

            captured_vote: list[dict] = []
            captured_defense: list[dict] = []
            done = False
            turn_id = 0
            while not done:
                player_id, observation = env.get_observation()
                phase = get_phase(env, observation)

                if is_voting(phase):
                    captured_vote.append({
                        "game_id": game_idx,
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                        "action_type": "vote",
                    })
                elif "discuss" in str(phase).lower():
                    captured_defense.append({
                        "game_id": game_idx,
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                        "action_type": "defense",
                    })

                if agents is not None:
                    action_out = agents[player_id](observation)
                    done, _ = env.step(action=action_out["action"])
                else:
                    done, _ = env.step(action="pass")
                turn_id += 1

            selected_vote = random.sample(captured_vote, min(len(captured_vote), situations_per_game)) if captured_vote else []
            selected_defense = list(captured_defense)
            selected_defense = random.sample(selected_defense, min(len(selected_defense), situations_per_game)) if selected_defense else []

            for row in selected_vote:
                row["action_type"] = "vote"
                if artifact_root is not None:
                    scenario_path, _ = _resolve_artifact_paths(row, "vote", artifact_root)
                    if scenario_path is not None:
                        scenario_path.write_text(str(row.get("observation", "")), encoding="utf-8")
                        row["scenario_path"] = str(scenario_path)
            selected_vote_rows.extend(selected_vote)

            for row in selected_defense:
                row["action_type"] = "defense"
                if artifact_root is not None:
                    scenario_path, _ = _resolve_artifact_paths(row, "defense", artifact_root)
                    if scenario_path is not None:
                        scenario_path.write_text(str(row.get("observation", "")), encoding="utf-8")
                        row["scenario_path"] = str(scenario_path)
            selected_defense_rows.extend(selected_defense)

        if llm is not None:
            llm = None
        if tokenizer is not None:
            tokenizer = None
        if agents is not None:
            agents = None
        gc.collect()
        torch.cuda.empty_cache()

        if selected_defense_rows:
            if defense_evaluator is None and judge_mode == "local":
                defense_evaluator = make_evaluator(judge_mode, judge_model, judge_gpu_memory_utilization)
            annotated_defense = []
            for row in selected_defense_rows:
                annotated_defense.append(
                    _annotate_defense_row(
                        row,
                        env,
                        evaluator=defense_evaluator,
                        judge_mode=judge_mode,
                        judge_model=judge_model,
                        gpu_memory_utilization=judge_gpu_memory_utilization,
                    )
                )
            selected_defense_rows = annotated_defense

        harvested.extend(selected_vote_rows)
        harvested.extend(selected_defense_rows)
    finally:
        if llm is not None:
            llm = None
        if tokenizer is not None:
            tokenizer = None
        if agents is not None:
            agents = None
        gc.collect()
        torch.cuda.empty_cache()

    debug_summary(harvested, "harvested_joint_observations")
    return harvested


class JointTrainDataset(Dataset):
    def __init__(self, samples: list[JointSample], tokenizer: AutoTokenizer, max_prompt_length: int = 1024, max_completion_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        phase = extract_phase(sample.observation)
        # Use the repository's designated prompt formatter for all agent turns.
        # This keeps the same game-state formatting as the rest of the pipeline.
        formatted_prompt = build_agent_prompt(sample.observation, phase)
        prompt_text = self.tokenizer.apply_chat_template(
            formatted_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        response_tokens = self.tokenizer(
            sample.response,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = torch.cat([
            prompt_tokens["input_ids"].squeeze(0),
            response_tokens["input_ids"].squeeze(0),
        ], dim=0)
        attention_mask = torch.cat([
            prompt_tokens["attention_mask"].squeeze(0),
            response_tokens["attention_mask"].squeeze(0),
        ], dim=0)
        labels = input_ids.clone()
        prompt_length = prompt_tokens["input_ids"].shape[1]
        labels[:prompt_length] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": torch.tensor(sample.advantage, dtype=torch.float32),
            "action_type": sample.action_type,
            "game_id": sample.game_id,
            "player_id": sample.player_id,
            "turn_id": sample.turn_id,
        }
        return item


def collate_joint(batch: list[dict]) -> dict:
    max_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    advantages = []
    action_types = []
    game_ids = []
    player_ids = []
    turn_ids = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad = max_length - seq_len
        input_ids.append(F.pad(item["input_ids"], (0, pad), value=0))
        attention_mask.append(F.pad(item["attention_mask"], (0, pad), value=0))
        labels.append(F.pad(item["labels"], (0, pad), value=-100))
        advantages.append(item["advantage"])
        action_types.append(item["action_type"])
        game_ids.append(item["game_id"])
        player_ids.append(item["player_id"])
        turn_ids.append(item["turn_id"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "advantages": torch.stack(advantages),
        "action_types": action_types,
        "game_ids": game_ids,
        "player_ids": player_ids,
        "turn_ids": turn_ids,
    }


def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)


def action_grpo_loss(
    policy_model: torch.nn.Module,
    old_policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    policy_log_probs = compute_sequence_log_probs(policy_model, input_ids, attention_mask, labels)
    with torch.no_grad():
        old_log_probs = compute_sequence_log_probs(old_policy_model, input_ids, attention_mask, labels)
        ref_log_probs = compute_sequence_log_probs(ref_model, input_ids, attention_mask, labels)

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

    log_ratio_ref = policy_log_probs - ref_log_probs
    kl_div = ((torch.exp(log_ratio_ref) - 1.0) - log_ratio_ref).mean()

    loss = policy_loss + kl_coef * kl_div
    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "total_loss": loss.item(),
        "ratio_mean": ratio.mean().item(),
        "advantage_mean": advantages.mean().item(),
    }
    return loss, metrics


class JointGRPOTrainer:
    def __init__(
        self,
        policy_model: torch.nn.Module,
        old_policy_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset: JointTrainDataset,
        output_dir: str,
        batch_size: int = 2,
        epochs: int = 1,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 10,
        save_steps: int = 100,
        lambda_vote: float = 1.0,
        lambda_defense: float = 1.0,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        bf16: bool = True,
    ):
        self.policy_model = policy_model
        self.old_policy_model = old_policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.lambda_vote = lambda_vote
        self.lambda_defense = lambda_defense
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.bf16 = bf16

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_joint,
            num_workers=0,
            pin_memory=True,
        )

    def _split_batch(self, batch: dict) -> tuple[list[dict], list[dict]]:
        vote_items = []
        defense_items = []
        for idx, action_type in enumerate(batch["action_types"]):
            if action_type == "vote":
                vote_items.append({
                    "input_ids": batch["input_ids"][idx].unsqueeze(0).cuda(),
                    "attention_mask": batch["attention_mask"][idx].unsqueeze(0).cuda(),
                    "labels": batch["labels"][idx].unsqueeze(0).cuda(),
                    "advantage": batch["advantages"][idx].unsqueeze(0).cuda(),
                })
            elif action_type == "defense":
                defense_items.append({
                    "input_ids": batch["input_ids"][idx].unsqueeze(0).cuda(),
                    "attention_mask": batch["attention_mask"][idx].unsqueeze(0).cuda(),
                    "labels": batch["labels"][idx].unsqueeze(0).cuda(),
                    "advantage": batch["advantages"][idx].unsqueeze(0).cuda(),
                })
        return vote_items, defense_items

    def _compute_combined_batch_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        vote_batch, defense_batch = self._split_batch(batch)

        vote_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
        defense_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
        metrics: dict[str, float] = {"vote": 0.0, "defense": 0.0, "total": 0.0}

        if vote_batch:
            vote_input_ids = torch.cat([item["input_ids"] for item in vote_batch], dim=0)
            vote_attention = torch.cat([item["attention_mask"] for item in vote_batch], dim=0)
            vote_labels = torch.cat([item["labels"] for item in vote_batch], dim=0)
            vote_advantages = torch.cat([item["advantage"] for item in vote_batch], dim=0)
            vote_loss, vote_metrics = action_grpo_loss(
                policy_model=self.policy_model,
                old_policy_model=self.old_policy_model,
                ref_model=self.ref_model,
                input_ids=vote_input_ids,
                attention_mask=vote_attention,
                labels=vote_labels,
                advantages=vote_advantages,
                clip_epsilon=self.clip_epsilon,
                kl_coef=self.kl_coef,
            )
            metrics["vote"] = vote_metrics["total_loss"]

        if defense_batch:
            def_input_ids = torch.cat([item["input_ids"] for item in defense_batch], dim=0)
            def_attention = torch.cat([item["attention_mask"] for item in defense_batch], dim=0)
            def_labels = torch.cat([item["labels"] for item in defense_batch], dim=0)
            def_advantages = torch.cat([item["advantage"] for item in defense_batch], dim=0)
            defense_loss, defense_metrics = action_grpo_loss(
                policy_model=self.policy_model,
                old_policy_model=self.old_policy_model,
                ref_model=self.ref_model,
                input_ids=def_input_ids,
                attention_mask=def_attention,
                labels=def_labels,
                advantages=def_advantages,
                clip_epsilon=self.clip_epsilon,
                kl_coef=self.kl_coef,
            )
            metrics["defense"] = defense_metrics["total_loss"]

        total_loss = (self.lambda_vote * vote_loss) + (self.lambda_defense * defense_loss)
        metrics["total"] = float(total_loss.item())
        return total_loss, metrics

    def train(self) -> None:
        self.policy_model.train()
        global_step = 0
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch + 1}/{self.epochs}] mixed vote + defense GRPO")
            progress = tqdm(self.train_loader, desc=f"Joint GRPO epoch {epoch + 1}")
            for step, batch in enumerate(progress):
                loss, metrics = self._compute_combined_batch_loss(batch)
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        print(
                            f"[Step {global_step}] total={metrics['total']:.4f} "
                            f"vote={metrics['vote']:.4f} defense={metrics['defense']:.4f}"
                        )

                    if global_step % self.save_steps == 0:
                        save_path = self.output_dir / f"checkpoint-{global_step}"
                        self.policy_model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)

            save_path = self.output_dir / "final"
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"[Saved] final mixed model to {save_path}")


def setup_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer


def setup_reference_model(model_name: str) -> AutoModelForCausalLM:
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model


def setup_old_policy_model(model_name: str) -> AutoModelForCausalLM:
    old_policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    old_policy.requires_grad_(False)
    old_policy.eval()
    return old_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed vote + defense GRPO training")
    parser.add_argument("--vote-data", type=Path, default=None, help="JSONL file produced by the dynamic vote loop")
    parser.add_argument("--defense-data", type=Path, default=None, help="JSONL file produced by the defense loop")
    parser.add_argument("--live-harvest", action="store_true", help="Harvest both vote and defense scenarios directly from self-play instead of reading static JSONL files.")
    parser.add_argument("--env-id", type=str, default="SecretMafia-v0")
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--games-per-iter", type=int, default=2)
    parser.add_argument("--situations-per-game", type=int, default=2)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Default actor model for training.")
    parser.add_argument("--old-policy-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Frozen model used to compute old-policy log-probs; defaults to the same Qwen2.5-7B base model.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "joint_dynamic_grpo")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory for saved observation and completion JSONL artifacts; defaults to <output-dir>/artifacts.")
    parser.add_argument("--judge-mode", choices=("local", "mock"), default="local")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct", help="Local judge model path or HF id")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lambda-vote", type=float, default=1.0)
    parser.add_argument("--lambda-defense", type=float, default=1.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    if not args.old_policy_model:
        args.old_policy_model = args.base_model

    artifact_root = args.artifacts_dir if args.artifacts_dir is not None else (Path(args.output_dir) / "artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    print(f"[Artifacts] joint loop debug files will be written to {artifact_root}")

    if args.live_harvest:
        harvested_rows = harvest_joint_observations(
            env_id=args.env_id,
            num_players=args.num_players,
            games_per_iter=args.games_per_iter,
            situations_per_game=args.situations_per_game,
            model_name=args.base_model,
            artifact_root=artifact_root,
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        rows = generate_joint_rollouts(
            model=args.base_model,
            rows=harvested_rows,
            args=args,
        )
        samples = build_joint_dataset(rows=rows)
    else:
        if args.vote_data is None or args.defense_data is None:
            raise ValueError("Either provide --vote-data and --defense-data or enable --live-harvest.")
        samples = build_joint_dataset(vote_path=args.vote_data, defense_path=args.defense_data)

    if not samples:
        raise ValueError("No mixed vote+defense samples available for training.")

    policy_model, tokenizer = setup_model(
        args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    old_policy_model = setup_old_policy_model(args.old_policy_model)
    ref_model = setup_reference_model(args.base_model)

    dataset = JointTrainDataset(
        samples,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    trainer = JointGRPOTrainer(
        policy_model=policy_model,
        old_policy_model=old_policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        output_dir=str(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lambda_vote=args.lambda_vote,
        lambda_defense=args.lambda_defense,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        bf16=args.bf16,
    )
    trainer.train()

    del trainer
    del policy_model
    del tokenizer
    del old_policy_model
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    print("[Done] joint dynamic GRPO training finished.")


if __name__ == "__main__":
    main()
