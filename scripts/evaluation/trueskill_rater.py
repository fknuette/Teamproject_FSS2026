"""TrueSkill rating update logic for per-role checkpoint ratings."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence, TYPE_CHECKING

import trueskill

if TYPE_CHECKING:
    from checkpoint_registry import CheckpointRegistry
    from run_eval_games import GameResult


def update_ratings_from_results(
    results: Sequence[GameResult],
    registry: CheckpointRegistry,
) -> None:
    """Update per-role TrueSkill ratings for all checkpoints in *results*.

    Processes each game result, updates in-memory ratings on *registry*, then
    atomically saves the registry to disk.
    """
    env = registry._build_trueskill_env()
    for game in results:
        updates = _compute_single_game(game, registry, env)
        for (ckpt_id, role), (new_mu, new_sigma, won) in updates.items():
            registry.update_rating(ckpt_id, role, new_mu, new_sigma, won)
    registry.save()


def _compute_single_game(
    game: GameResult,
    registry: CheckpointRegistry,
    env: trueskill.TrueSkill,
) -> dict[tuple[str, str], tuple[float, float, bool]]:
    """Compute updated TrueSkill ratings for a single game.

    Groups players by team (Mafia vs Village) so Village roles cooperate rather
    than compete. Per-role ratings are maintained: each player's role-specific
    rating is used for lookup and written back after the update.

    Returns a dict mapping *(checkpoint_id, role)* to *(new_mu, new_sigma, won)*.
    Duplicate checkpoints in the same role are averaged.
    """
    winning_team = _winning_team(game)
    mafia_players = [p for p in game.players if p.role == "Mafia"]
    village_players = [p for p in game.players if p.role != "Mafia"]

    if not mafia_players or not village_players:
        return {}

    mafia_ratings = [registry.get_trueskill_rating(p.checkpoint, p.role) for p in mafia_players]
    village_ratings = [registry.get_trueskill_rating(p.checkpoint, p.role) for p in village_players]

    mafia_rank = 0 if winning_team == "Mafia" else 1
    village_rank = 1 if winning_team == "Mafia" else 0
    updated_mafia, updated_village = env.rate(
        [mafia_ratings, village_ratings], ranks=[mafia_rank, village_rank]
    )

    # Collect updates; average duplicates within the same (checkpoint_id, role)
    accum: dict[tuple[str, str], list[tuple[float, float, bool]]] = defaultdict(list)
    mafia_won = winning_team == "Mafia"
    for player, new_rating in zip(mafia_players, updated_mafia):
        accum[(player.checkpoint, player.role)].append((new_rating.mu, new_rating.sigma, mafia_won))
    for player, new_rating in zip(village_players, updated_village):
        accum[(player.checkpoint, player.role)].append((new_rating.mu, new_rating.sigma, not mafia_won))

    result: dict[tuple[str, str], tuple[float, float, bool]] = {}
    for (ckpt_id, role), entries in accum.items():
        avg_mu = sum(e[0] for e in entries) / len(entries)
        avg_sigma = sum(e[1] for e in entries) / len(entries)
        result[(ckpt_id, role)] = (avg_mu, avg_sigma, entries[0][2])

    return result


def _winning_team(game: GameResult) -> str:
    """Return ``"Mafia"`` or ``"Village"`` based on the game result."""
    return game.winning_team
