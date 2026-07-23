"""Matchmaking engine for evaluation games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for an agent in a matchup."""
    
    checkpoint_id: str  # Model checkpoint ID
    player_idx: int  # Player position (0-5)
    team: str  # "Mafia" or "Village"
    role: str  # "Mafia" or "Villager"


@dataclass(frozen=True)
class Matchup:
    """Represents a matchup between two teams with specific roles."""
    
    agents: tuple  # All 6 agents with their roles


class Matchmaker(ABC):
    """Abstract base class for matchmaking strategies."""
    
    @abstractmethod
    def get_matchups(
        self,
        eval_checkpoint: str,
        available_checkpoints: List[str] | None = None,
    ) -> dict:
        """Generate matchups for evaluation."""
        pass


class SimplePairMatchmaker(Matchmaker):
    """Matches eval_checkpoint vs a single baseline checkpoint."""
    
    # Game configuration - modify here to change evaluation settings
    NUM_WEREWOLVES = 2
    NUM_VILLAGERS = 4
    NUM_GAMES_EVAL_MAFIA = 2
    NUM_GAMES_EVAL_VILLAGER = 2
    
    def __init__(self, baseline_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.baseline_checkpoint = baseline_checkpoint
    
    def get_matchups(
        self,
        eval_checkpoint: str,
        available_checkpoints: List[str] | None = None,
    ) -> dict:
        """Generate matchups: eval vs baseline with role separation."""
        matchups_dict = {}
        
        # Matchup 1: eval as Mafia, baseline as Villagers
        if self.NUM_GAMES_EVAL_MAFIA > 0:
            agents_mafia = []
            for i in range(self.NUM_WEREWOLVES):
                agents_mafia.append(
                    AgentConfig(
                        checkpoint_id=eval_checkpoint,
                        player_idx=i,
                        team="Mafia",
                        role="Mafia",
                    )
                )
            for i in range(self.NUM_VILLAGERS):
                agents_mafia.append(
                    AgentConfig(
                        checkpoint_id=self.baseline_checkpoint,
                        player_idx=self.NUM_WEREWOLVES + i,
                        team="Village",
                        role="Village",
                    )
                )
            
            matchup_mafia = Matchup(agents=tuple(agents_mafia))
            matchups_dict[matchup_mafia] = self.NUM_GAMES_EVAL_MAFIA
        
        # Matchup 2: eval as Villagers, baseline as Werewolves
        if self.NUM_GAMES_EVAL_VILLAGER > 0:
            agents_villager = []
            for i in range(self.NUM_WEREWOLVES):
                agents_villager.append(
                    AgentConfig(
                        checkpoint_id=self.baseline_checkpoint,
                        player_idx=i,
                        team="Mafia",
                        role="Mafia",
                    )
                )
            for i in range(self.NUM_VILLAGERS):
                agents_villager.append(
                    AgentConfig(
                        checkpoint_id=eval_checkpoint,
                        player_idx=self.NUM_WEREWOLVES + i,
                        team="Village",
                        role="Village",
                    )
                )
            
            matchup_villager = Matchup(agents=tuple(agents_villager))
            matchups_dict[matchup_villager] = self.NUM_GAMES_EVAL_VILLAGER
        
        return matchups_dict
