"""Matchmaking engine for evaluation games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import random
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from checkpoint_registry import CheckpointRegistry

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
    

class RandomMatchmaker(Matchmaker):
    """Generates matchups by randomly sampling opponents from a registry.

    For each evaluation run, ``min_games_per_team_role`` games are scheduled
    with the eval checkpoint playing as Mafia, and another
    ``min_games_per_team_role`` games with it playing as Village.  Opponents
    are drawn uniformly at random (with replacement) from the registry.

    If the registry contains only the eval checkpoint, self-play is used.
    """

    NUM_MAFIA = 2   # player slots assigned team="Mafia"
    NUM_VILLAGE = 4  # player slots assigned team="Village" (sub-roles set at runtime)

    def __init__(
        self,
        registry: CheckpointRegistry,
        min_games_per_team_role: int = 10,
        seed: int | None = None,
    ) -> None:
        self._registry = registry
        self._min_games = min_games_per_team_role
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Matchmaker interface
    # ------------------------------------------------------------------

    def get_matchups(
        self,
        eval_checkpoint: str,
        available_checkpoints: List[str] | None = None,
    ) -> dict[Matchup, int]:
        """Return a dict mapping each ``Matchup`` to its game count."""
        opponent_pool = self._get_opponent_pool(eval_checkpoint, available_checkpoints)
        matchups: dict[Matchup, int] = {}

        for eval_team in ("Mafia", "Village"):
            for _ in range(self._min_games):
                matchup = self._build_matchup(eval_checkpoint, eval_team, opponent_pool)
                matchups[matchup] = matchups.get(matchup, 0) + 1

        return matchups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matchup(
        self,
        eval_checkpoint: str,
        eval_team: str,
        opponent_pool: list[str],
    ) -> Matchup:
        """Build a single ``Matchup`` with eval occupying one slot on *eval_team*.

        Slot layout: idx 0-1 = Mafia team, idx 2-5 = Village team.
        Eval takes slot 0 (Mafia) or slot 2 (first Village slot); all other
        slots are independently sampled from *opponent_pool*.
        The game engine overwrites Village role with the actual sub-role.
        """
        agents: list[AgentConfig] = []

        if eval_team == "Mafia":
            # eval in slot 0; remaining Mafia slot(s) filled by opponents
            agents.append(AgentConfig(checkpoint_id=eval_checkpoint, player_idx=0, team="Mafia", role="Mafia"))
            for i, opp in enumerate(self._rng.choices(opponent_pool, k=self.NUM_MAFIA - 1), start=1):
                agents.append(AgentConfig(checkpoint_id=opp, player_idx=i, team="Mafia", role="Mafia"))
            # all Village slots filled by opponents
            for j, opp in enumerate(self._rng.choices(opponent_pool, k=self.NUM_VILLAGE)):
                agents.append(AgentConfig(checkpoint_id=opp, player_idx=self.NUM_MAFIA + j, team="Village", role="Village"))
        else:  # eval_team == "Village"
            # all Mafia slots filled by opponents
            for i, opp in enumerate(self._rng.choices(opponent_pool, k=self.NUM_MAFIA)):
                agents.append(AgentConfig(checkpoint_id=opp, player_idx=i, team="Mafia", role="Mafia"))
            # eval in first Village slot; remaining Village slots filled by opponents
            agents.append(AgentConfig(checkpoint_id=eval_checkpoint, player_idx=self.NUM_MAFIA, team="Village", role="Village"))
            for j, opp in enumerate(self._rng.choices(opponent_pool, k=self.NUM_VILLAGE - 1), start=1):
                agents.append(AgentConfig(checkpoint_id=opp, player_idx=self.NUM_MAFIA + j, team="Village", role="Village"))

        return Matchup(agents=tuple(agents))

    def _get_opponent_pool(
        self,
        eval_checkpoint: str,
        available_checkpoints: list[str] | None,
    ) -> list[str]:
        """Return the list of checkpoints to sample opponents from.

        Excludes eval_checkpoint when other options exist.  Falls back to
        self-play (eval_checkpoint only) when the registry has a single entry.
        """
        if available_checkpoints is not None:
            pool: list[str] = []
            for c in available_checkpoints:
                pool.append(c)
        else:
            # Use the registered checkpoint paths
            pool = list(self._registry.all_ids())

        others = [c for c in pool if c != eval_checkpoint]
        return others if others else [eval_checkpoint]
