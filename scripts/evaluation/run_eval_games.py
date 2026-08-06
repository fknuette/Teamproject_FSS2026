"""Evaluation games orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import textarena as ta

if TYPE_CHECKING:
    from checkpoint_registry import CheckpointRegistry
    from self_play_textarena import VLLMTextArenaAgent


@dataclass
class PlayerInfo:
    """Information about a player in an eval game."""
    player_id: int
    checkpoint: str  # checkpoint_id
    team: str  # "Mafia" or "Village"
    role: str  # "Mafia" or "Villager"


@dataclass
class GameResult:
    """Result of an eval game."""
    game_id: int
    winning_team: str  # "Mafia" or "Village"
    players: list[PlayerInfo]  # All players with their info


def run_eval_games(
    matchups_dict: dict,
    output_path: Path,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.6,
    registry: CheckpointRegistry | None = None,
) -> list[GameResult]:
    """Run evaluation games with different agents per player.
    
    Args:
        matchups_dict: Dict mapping Matchup objects to number of games.
                      Matchup contains AgentConfig for each player with role/team info.
        output_path: Path to save game results JSON.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: GPU memory utilization ratio for vLLM.
    """
    from self_play_textarena import _simulate_game
    from matchmaker import Matchup, AgentConfig
    from agent_factory import AgentFactory
    
    env_id = "SecretMafia-v0"  # Fixed environment ID (use registered textarena env)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent factory
    factory = AgentFactory(
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    all_results: list[GameResult] = []
    global_game_id = 0
    
    # Iterate over each matchup and number of games
    for matchup, num_games in matchups_dict.items():
        # Build agent mapping: player_id -> (checkpoint_id, agent, team, role)
        agent_configs: dict[int, tuple[str, VLLMTextArenaAgent, str, str]] = {}
        
        for agent_config in matchup.agents:
            checkpoint_path = (
                registry.get(agent_config.checkpoint_id).path
                if registry is not None and agent_config.checkpoint_id in registry.all_ids()
                else agent_config.checkpoint_id
            )
            agent = factory.create_agent(checkpoint_path)
            agent_configs[agent_config.player_idx] = (
                agent_config.checkpoint_id,
                agent,
                agent_config.team,
                agent_config.role,
            )
        
        num_players = max(agent_configs) + 1

        # Play num_games games for this matchup
        for game_in_matchup in range(num_games):
            reward_map, _, actual_player_info = _simulate_game(global_game_id, agent_configs, env_id, num_players=num_players, is_eval=True)

            # Determine winning team from actual TextArena role assignments
            mafia_players = {pid for pid, (_, team, _) in actual_player_info.items() if team == "Mafia"}
            villager_players = {pid for pid, (_, team, _) in actual_player_info.items() if team == "Village"}
            mafia_reward = sum(reward_map.get(pid, 0.0) for pid in mafia_players)
            villager_reward = sum(reward_map.get(pid, 0.0) for pid in villager_players)
            winning_team = "Mafia" if mafia_reward > villager_reward else "Village"

            # Create player info using actual TextArena player assignments
            players_info = []
            for player_id, (checkpoint_id, team, role) in actual_player_info.items():
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
    return all_results
