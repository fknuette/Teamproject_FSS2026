from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import textarena as ta
import re
from textarena.core import Agent
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from argument_parser import build_parser
from teamproject_fss2026.textarena_utils import build_agent_prompt, parse_model_response


@dataclass
class TurnRecord:
    game_id: int
    observation: str
    response: str
    reward: float = 0.0
    player_id: int = 0
    turn_id: int = 0


@dataclass
class PlayerInfo:
    """Information about a player in an eval game."""
    player_id: int
    checkpoint: str  # checkpoint_id
    team: str  # "team_a" or "team_b"
    role: str  # "werewolf" or "villager"


@dataclass
class GameResult:
    """Result of an eval game."""
    game_id: int
    winning_team: str  # "team_a" or "team_b"
    players: list[PlayerInfo]  # All 6 players with their info


class VLLMTextArenaAgent(Agent):
    def __init__(self, llm: LLM, tokinizer: AutoTokenizer) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokinizer

    def __call__(self, observation: str) -> dict[str, str]:
        
        # Findout in we will vote or not
        matches = re.findall(r'\[GAME\](.*)(?=\n|$)', observation)
        valid_matches = [m.strip() for m in matches if "invalid move" not in m.lower()]
        phase_text = valid_matches[-1]
        if "Voting phase" in phase_text:
            phase = "Voting"
        elif "Discuss" in phase_text:
            phase = "Discuss"
        else:
            phase = "Action"

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
            env = ta.make(
                args.env_id,
                mafia_ratio=args.num_mafia / args.num_players,
            )
            env.reset(num_players=args.num_players)

            agents: dict[int, VLLMTextArenaAgent] = {
                pid: VLLMTextArenaAgent(llm, tokenizer)
                for pid in range(args.num_players)
            }

            game_records: list[TurnRecord] = []
            done = False
            turn_id = 0

            # ===== INVALID MOVE HANDLING START =====
            invalid_turn_ids: set[int] = set()
            seen_eliminated_lines_by_player: dict[int, set[str]] = {pid: set() for pid in agents}
            # ===== INVALID MOVE HANDLING END =====

            while not done:
                player_id, observation = env.get_observation()

                # ===== INVALID MOVE HANDLING START =====
                _update_invalid_turn_tracking(
                    observation=observation,
                    player_id=player_id,
                    game_records=game_records,
                    invalid_turn_ids=invalid_turn_ids,
                    seen_eliminated_lines_by_player=seen_eliminated_lines_by_player,
                )
                # ===== INVALID MOVE HANDLING END =====

                agent_out = agents[player_id](observation)
                done, _ = env.step(action=agent_out["action"])

                game_records.append(
                    TurnRecord(
                        game_id=game_id,
                        turn_id=turn_id,
                        player_id=player_id,
                        observation=observation,
                        response=agent_out["response"],
                    )
                )
                turn_id += 1

            rewards, _game_info = env.close()
            reward_map = _normalize_rewards(rewards)

            for record in game_records:
                record.reward = reward_map.get(record.player_id, 0.0)

                # ===== INVALID MOVE HANDLING START =====
                if record.turn_id in invalid_turn_ids:
                    record.reward = -1.0
                # ===== INVALID MOVE HANDLING END =====

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


def run_eval_games(
    matchups_dict: dict,
    output_path: Path,
    tensor_parallel_size: int = 1,
) -> None:
    """Run evaluation games with different agents per player.
    
    Args:
        matchups_dict: Dict mapping Matchup objects to number of games.
                      Matchup contains AgentConfig for each of 6 players with role/team info.
        output_path: Path to save game results JSON.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
    """
    from evaluation.matchmaker import Matchup, AgentConfig
    from evaluation.agent_factory import AgentFactory
    
    env_id = "Mafia"  # Fixed environment ID
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent factory
    factory = AgentFactory(tensor_parallel_size=tensor_parallel_size)
    
    all_results: list[GameResult] = []
    global_game_id = 0
    
    # Iterate over each matchup and number of games
    for matchup, num_games in matchups_dict.items():
        # Build agent mapping: player_id -> (checkpoint_id, agent, team, role)
        agent_configs: dict[int, tuple[str, VLLMTextArenaAgent, str, str]] = {}
        
        for agent_config in matchup.agents:
            agent = factory.create_agent(agent_config.checkpoint_id)
            agent_configs[agent_config.player_idx] = (
                agent_config.checkpoint_id,
                agent,
                agent_config.team,
                agent_config.role,
            )
        
        # Determine team assignments
        team_a_players = set(
            idx for idx, (_, _, team, _) in agent_configs.items() if team == "team_a"
        )
        team_b_players = set(
            idx for idx, (_, _, team, _) in agent_configs.items() if team == "team_b"
        )
        
        # Play num_games games for this matchup
        for game_in_matchup in range(num_games):
            env = ta.make(env_id)
            env.reset(num_players=6)
            
            agents: dict[int, VLLMTextArenaAgent] = {}
            for player_id in range(6):
                _, agent, _, _ = agent_configs[player_id]
                agents[player_id] = agent
            
            done = False
            
            # ===== INVALID MOVE HANDLING START =====
            invalid_turn_ids: set[int] = set()
            seen_eliminated_lines_by_player: dict[int, set[str]] = {pid: set() for pid in agents}
            # ===== INVALID MOVE HANDLING END =====
            
            turn_id = 0
            
            while not done:
                player_id, observation = env.get_observation()
                
                # ===== INVALID MOVE HANDLING START =====
                _update_invalid_turn_tracking(
                    observation=observation,
                    player_id=player_id,
                    game_records=[],
                    invalid_turn_ids=invalid_turn_ids,
                    seen_eliminated_lines_by_player=seen_eliminated_lines_by_player,
                )
                # ===== INVALID MOVE HANDLING END =====
                
                agent_out = agents[player_id](observation)
                done, _ = env.step(action=agent_out["action"])
                turn_id += 1
            
            rewards, game_info = env.close()
            reward_map = _normalize_rewards(rewards)
            
            # Determine winning team based on rewards
            team_a_reward = sum(reward_map.get(pid, 0.0) for pid in team_a_players)
            team_b_reward = sum(reward_map.get(pid, 0.0) for pid in team_b_players)
            winning_team = "team_a" if team_a_reward > team_b_reward else "team_b"
            
            # Create player info for all players
            players_info = []
            for player_id in range(6):
                checkpoint_id, _, team, role = agent_configs[player_id]
                players_info.append(
                    PlayerInfo(
                        player_id=player_id,
                        checkpoint=checkpoint_id,
                        team=team,
                        role=role,
                    )
                )
            
            result = GameResult(
                game_id=global_game_id,
                winning_team=winning_team,
                players=players_info,
            )
            
            all_results.append(result)
            global_game_id += 1
    
    # Save all results to single file
    with output_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(all_results)} eval game results to {output_path}")


def main() -> None:
    """CLI entry point for self-play trace collection."""
    parser = build_parser(context="self_play")
    args = parser.parse_args()
    run_self_play(args)


if __name__ == "__main__":
    main()
