from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import textarena as ta
from textarena.core import Agent
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from argument_parser import build_parser
from teamproject_fss2026.textarena_utils import build_agent_prompt, parse_model_response, extract_phase


@dataclass
class TurnRecord:
    game_id: int
    observation: str
    response: str
    reward: float = 0.0
    player_id: int = 0
    turn_id: int = 0


class VLLMTextArenaAgent(Agent):
    def __init__(self, llm: LLM, tokinizer: AutoTokenizer) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokinizer

    @property
    def model_name(self) -> str:
        """Extrahiert den geladenen Modellpfad/Namen aus vLLM."""
        try:
            return self.llm.llm_engine.model_config.model
        except AttributeError:
            return "Unbekanntes vLLM Modell"

    def __call__(self, observation: str) -> dict[str, str]:
        
        phase = extract_phase(observation)

        # Tag-Modus: Darf reden, stoppt nur am nächsten Block-Trenner
        if phase == "Discuss":
            current_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=200,
                # stop=["###"] # Stoppt erst am nächsten Block
            )
        else:
            # Logik-Check: Nacht (Nur Nummer) vs. Tag (Reden/Rechnen)
            current_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=100,
                # stop=[".", "\n", "]", " "] # Stoppt sofort nach der Zahl/Klammer
            )
            # Hier evtl. temperature > 0 lassen für natürlichere Sprache
        
        
        own_prompt = build_agent_prompt(observation, phase)
        
        
        prompt = self.tokenizer.apply_chat_template(
                    own_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
        
        outputs = self.llm.generate([prompt], current_params)
        raw_text = outputs[0].outputs[0].text
        parsed = parse_model_response(raw_text)
        return {
            "response": raw_text,
            "action": parsed.action,
        }


def _normalize_rewards(rewards: Any) -> dict[int, float]:
    if isinstance(rewards, dict):
        return {int(k): float(v) for k, v in rewards.items()}
    if isinstance(rewards, (list, tuple)):
        return {i: float(v) for i, v in enumerate(rewards)}
    raise ValueError(f"Unsupported rewards format: {type(rewards)}")


def _simulate_game(
    game_id: int,
    agents_or_config: Union[dict[int, VLLMTextArenaAgent], dict[int, tuple[str, VLLMTextArenaAgent, str, str]]],
    env_id: str,
    num_players: int = 8,
    num_mafia: int = 2,
    is_eval: bool = False,
) -> tuple[dict[int, float], list[TurnRecord]]:
    """Run one game loop and return rewards plus turn records."""
    env = ta.make(env_id, mafia_ratio=args.num_mafia / args.num_players)
    env.reset(num_players=num_players)
    game_records: list[TurnRecord] = []
    invalid_turn_ids: set[int] = set()
    seen_eliminated_lines_by_player: dict[int, set[str]] = {pid: set() for pid in agents_or_config}
    done = False
    turn_id = 0

    actual_player_info: dict[int, tuple[str, str, str]] = {}  # player_id -> (checkpoint, team, role)
    if is_eval:
        ta_assigned_roles = getattr(env, "roles", {})
        print(f"Assigned roles: {ta_assigned_roles}")
        agents: dict[int, VLLMTextArenaAgent] = {}
        config_team_a = [(cfg[0], cfg[1]) for cfg in agents_or_config.values() if cfg[2] == "Mafia"]
        config_team_b = [(cfg[0], cfg[1]) for cfg in agents_or_config.values() if cfg[2] == "Village"]

        for ta_player_id, ta_role in ta_assigned_roles.items():
            role_name = type(ta_role).__name__
            team = "Mafia" if role_name == "Mafia" else "Village"
            if team == "Mafia":
                checkpoint, agent_instance = config_team_a.pop(0)
            else:
                checkpoint, agent_instance = config_team_b.pop(0)
            agents[ta_player_id] = agent_instance
            actual_player_info[ta_player_id] = (checkpoint, team, role_name)

        for pid, (checkpoint, team, role_name) in actual_player_info.items():
            env_role = type(env.roles[pid]).__name__
            agent_model = agents[pid].model_name
            role_ok = "✓" if env_role == role_name else f"✗ MISMATCH (env={env_role})"
            model_ok = "✓" if agent_model == checkpoint else f"✗ MISMATCH (agent={agent_model})"
            print(f"▶️ Player {pid:2} | Role: {role_name:<10} {role_ok} | Model: {checkpoint} {model_ok}")

    else:
        agents = agents_or_config  # Use the provided agents directly
    
    
    while not done:
        player_id, observation = env.get_observation()
        _update_invalid_turn_tracking(
            observation=observation,
            player_id=player_id,
            game_records=game_records,
            invalid_turn_ids=invalid_turn_ids,
            seen_eliminated_lines_by_player=seen_eliminated_lines_by_player,
        )

        agent_out = agents[player_id](observation)
        done, _ = env.step(action=agent_out["action"])

        game_records.append(
            TurnRecord(
                game_id=game_id,
                observation=observation,
                response=agent_out["response"],
                player_id=player_id,
                turn_id=turn_id,
            )
        )
        turn_id += 1

    rewards, _game_info = env.close()
    reward_map = _normalize_rewards(rewards)

    for record in game_records:
        record.reward = reward_map.get(record.player_id, 0.0)
        if record.turn_id in invalid_turn_ids:
            record.reward = -1.0

    return reward_map, game_records, actual_player_info


# ===== INVALID MOVE HANDLING START =====

ATTEMPTED_INVALID_AT_END_RE = re.compile(
    r"\[GAME\] Player (?P<pid>\d+) attempted an invalid move\. "
    r"Reason: .*?Please resubmit a valid move and remember to follow the game rules to avoid penalties\.\s*$",
    re.DOTALL,
)

ELIMINATED_INVALID_RE = re.compile(
    r"^\[GAME\] Player (?P<pid>\d+) has been eliminated by making an invalid move\.\.?$"
)


def _extract_invalid_pid_from_attempted_at_end(observation: str) -> int | None:
    match = ATTEMPTED_INVALID_AT_END_RE.search(observation)
    return int(match.group("pid")) if match else None


def _extract_eliminated_invalid_lines(observation: str) -> set[str]:
    return {
        line.strip()
        for line in observation.splitlines()
        if line.strip() and ELIMINATED_INVALID_RE.match(line.strip())
    }


def _extract_invalid_pid_from_eliminated_line(line: str) -> int | None:
    match = ELIMINATED_INVALID_RE.match(line)
    return int(match.group("pid")) if match else None


def _mark_last_turn_invalid(
    game_records: list[TurnRecord],
    invalid_turn_ids: set[int],
    player_id: int,
) -> None:
    for rec in reversed(game_records):
        if rec.player_id == player_id:
            invalid_turn_ids.add(rec.turn_id)
            return


def _update_invalid_turn_tracking(
    observation: str,
    player_id: int,
    game_records: list[TurnRecord],
    invalid_turn_ids: set[int],
    seen_eliminated_lines_by_player: dict[int, set[str]],
) -> None:
    attempted_invalid_pid = _extract_invalid_pid_from_attempted_at_end(observation)
    if attempted_invalid_pid is not None:
        _mark_last_turn_invalid(game_records, invalid_turn_ids, attempted_invalid_pid)

    current_eliminated_lines = _extract_eliminated_invalid_lines(observation)
    new_eliminated_lines = current_eliminated_lines - seen_eliminated_lines_by_player[player_id]

    for line in new_eliminated_lines:
        eliminated_pid = _extract_invalid_pid_from_eliminated_line(line)
        if eliminated_pid is not None:
            _mark_last_turn_invalid(game_records, invalid_turn_ids, eliminated_pid)

    seen_eliminated_lines_by_player[player_id] = current_eliminated_lines

# ===== INVALID MOVE HANDLING END =====


def run_self_play(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not 6 <= args.num_players <= 15:
        raise ValueError("num_players must be between 6 and 15")
    if not 1 <= args.num_mafia <= args.num_players - 2:
        raise ValueError("num_mafia must leave room for one Doctor and one Detective")

    # Accept either `--model-a` (from online loop) or `--model` (self-play CLI)
    model_name = getattr(args, "model", None)
    if model_name is None:
        raise ValueError("No model specified: provide --model or --model-a")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.6),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        all_records: list[TurnRecord] = []

        for game_id in range(args.num_games):
            agents: dict[int, VLLMTextArenaAgent] = {
                player_id: VLLMTextArenaAgent(llm, tokenizer)
                for player_id in range(args.num_players)
            }

        _, game_records, _ = _simulate_game(game_id, agents, args.env_id, num_players=args.num_players, num_mafia=args.num_mafia, is_eval=False)
        all_records.extend(game_records)

            with output_path.open("w", encoding="utf-8") as f:
                for rec in all_records:
                    f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            print(f"Wrote {len(all_records)} turn records to {output_path}")
    finally:
        close_fn = getattr(llm, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        del llm
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    """CLI entry point for self-play trace collection."""
    parser = build_parser(context="self_play")
    args = parser.parse_args()
    run_self_play(args)


if __name__ == "__main__":
    main()
