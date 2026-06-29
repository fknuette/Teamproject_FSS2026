"""Matchmaking engine for evaluation games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AgentConfig:
    """Configuration for an agent in a matchup."""
    
    checkpoint_id: str  # Model checkpoint ID
    player_idx: int  # Player position (0-5)
    team: str  # "team_a" or "team_b"
    role: str  # "werewolf" or "villager"


@dataclass
class Matchup:
    """Represents a matchup between two teams with specific roles.
    
    A matchup defines the team composition and roles for a set of games.
    The number of games to play with this matchup is provided separately
    by the matchmaker as dict[Matchup, int].
    """
    
    agents: List[AgentConfig]  # All 6 agents with their roles


class Matchmaker(ABC):
    """Abstract base class for matchmaking strategies."""
    
    @abstractmethod
    def get_matchups(
        self,
        eval_checkpoint: str,
        available_checkpoints: List[str] | None = None,
    ) -> dict[Matchup, int]:
        """Generate matchups for evaluation.
        
        Args:
            eval_checkpoint: The checkpoint being evaluated.
            available_checkpoints: Optional list of available checkpoints.
            
        Returns:
            Dict mapping Matchup objects to number of games to play.
        """
        pass


class SimplePairMatchmaker(Matchmaker):
    """Matches eval_checkpoint vs a single baseline checkpoint.
    
    Strategy:
    - Eval checkpoint always forms a complete team (either all Werewolves or all Villagers)
    - Baseline checkpoint forms the opposing team
    - Fixed game configuration (change in code if needed)
    
    Team composition (6 players):
    - Werewolf team: 2 players
    - Villager team: 4 players
    """
    
    # Game configuration - modify here to change evaluation settings
    NUM_WEREWOLVES = 2
    NUM_VILLAGERS = 4
    NUM_GAMES_EVAL_WEREWOLF = 50
    NUM_GAMES_EVAL_VILLAGER = 50
    
    def __init__(
        self,
        baseline_checkpoint: str = "Qwen/Qwen3-8B",
    ):
        """Initialize SimplePairMatchmaker.
        
        Args:
            baseline_checkpoint: Checkpoint to play against.
        
        To change game configuration, modify class constants:
        - NUM_WEREWOLVES
        - NUM_VILLAGERS
        - NUM_GAMES_EVAL_WEREWOLF
        - NUM_GAMES_EVAL_VILLAGER
        """
        self.baseline_checkpoint = baseline_checkpoint
    
    def get_matchups(
        self,
        eval_checkpoint: str,
        available_checkpoints: list[str] | None = None,
    ) -> dict[Matchup, int]:
        """Generate matchups: eval vs baseline with role separation.
        
        Returns dict with matchups and game counts based on class configuration.
        """
        matchups_dict = {}
        
        # Matchup 1: eval as Werewolves, baseline as Villagers
        if self.NUM_GAMES_EVAL_WEREWOLF > 0:
            agents_werewolf = []
            # Eval werewolves
            for i in range(self.NUM_WEREWOLVES):
                agents_werewolf.append(
                    AgentConfig(
                        checkpoint_id=eval_checkpoint,
                        player_idx=i,
                        team="team_a",
                        role="werewolf",
                    )
                )
            # Baseline villagers
            for i in range(self.NUM_VILLAGERS):
                agents_werewolf.append(
                    AgentConfig(
                        checkpoint_id=self.baseline_checkpoint,
                        player_idx=self.NUM_WEREWOLVES + i,
                        team="team_b",
                        role="villager",
                    )
                )
            
            matchup_werewolf = Matchup(agents=agents_werewolf)
            matchups_dict[matchup_werewolf] = self.NUM_GAMES_EVAL_WEREWOLF
        
        # Matchup 2: eval as Villagers, baseline as Werewolves
        if self.NUM_GAMES_EVAL_VILLAGER > 0:
            agents_villager = []
            # Baseline werewolves
            for i in range(self.NUM_WEREWOLVES):
                agents_villager.append(
                    AgentConfig(
                        checkpoint_id=self.baseline_checkpoint,
                        player_idx=i,
                        team="team_a",
                        role="werewolf",
                    )
                )
            # Eval villagers
            for i in range(self.NUM_VILLAGERS):
                agents_villager.append(
                    AgentConfig(
                        checkpoint_id=eval_checkpoint,
                        player_idx=self.NUM_WEREWOLVES + i,
                        team="team_b",
                        role="villager",
                    )
                )
            
            matchup_villager = Matchup(agents=agents_villager)
            matchups_dict[matchup_villager] = self.NUM_GAMES_EVAL_VILLAGER
        
        return matchups_dict
