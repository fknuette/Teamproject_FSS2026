"""TrueSkill rating update logic for per-role checkpoint ratings."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence, TYPE_CHECKING
from statistics import NormalDist
import trueskill

_ROLE_BIAS_SIGMA = 1e-6

if TYPE_CHECKING:
    from checkpoint_registry import CheckpointRegistry
    from run_eval_games import GameResult


def _apply_role_performance_offset(rating: trueskill.Rating, role: str) -> trueskill.Rating:
    """Return a temporary copy with the role-based performance offset applied.

    The offset is meant only for evaluation-time team balancing and must never be
    written back into the persistent registry.
    """
    if role == "Mafia":
        return trueskill.Rating(mu=rating.mu + _ROLE_BIAS_MU, sigma=rating.sigma)
    if role == "Villager":
        return trueskill.Rating(mu=rating.mu - _ROLE_BIAS_MU, sigma=rating.sigma)
    return trueskill.Rating(mu=rating.mu, sigma=rating.sigma)


def _remove_role_performance_offset(rating: trueskill.Rating, role: str) -> trueskill.Rating:
    """Undo the temporary role-performance offset from a copied rating."""
    if role == "Mafia":
        return trueskill.Rating(mu=rating.mu - _ROLE_BIAS_MU, sigma=rating.sigma)
    if role == "Villager":
        return trueskill.Rating(mu=rating.mu + _ROLE_BIAS_MU, sigma=rating.sigma)
    return trueskill.Rating(mu=rating.mu, sigma=rating.sigma)


def _virtual_role_bias_rating(
    all_ratings: Sequence[trueskill.Rating],
    env: trueskill.TrueSkill,
    n_mafia: int = 2,
    n_villager: int = 6,
    p_mafia: float = 0.75,
) -> trueskill.Rating:
    """Return a frozen virtual team member representing the Mafia advantage.

    The variance term uses the squared uncertainties of all players in the match,
    not only the Mafia team, as required by the formula:
    ``mu_v = 4 * mu_0 + 0.6745 * sqrt(9 * beta^2 + sum_i sigma_i^2)``.

    The bias is intentionally *not* stored in the persistent rating; it only
    influences the match update so the model ratings remain role-independent.
    """
    n_total = n_mafia + n_villager + 1
    var = (
        sum(r.sigma**2 for r in all_ratings)
        + n_total * (env.beta**2 + env.tau**2)
        + _ROLE_BIAS_SIGMA**2
    )
    z = NormalDist().inv_cdf(p_mafia)
    #mu_0 = float(getattr(env, "mu", 25.0))
    #beta = float(getattr(env, "beta", 25.0 / 6.0))
    mu_v = (n_villager - n_mafia) * env.mu + z * math.sqrt(var)
    return trueskill.Rating(mu=mu_v, sigma=_ROLE_BIAS_SIGMA)


def update_ratings_from_results(
    results: Sequence[GameResult],
    registry: CheckpointRegistry,
    fixed_baseline_checkpoint: str | None = None,
) -> None:
    """Update per-role TrueSkill ratings for all checkpoints in *results*.

    Processes each game result, updates in-memory ratings on *registry*, then
    atomically saves the registry to disk.

    If *fixed_baseline_checkpoint* is provided, ratings for that checkpoint are
    left unchanged so the baseline remains a stable reference point.
    """
    env = registry._build_trueskill_env()
    for game in results:
        updates = _compute_single_game(game, registry, env)
        for (ckpt_id, role), (new_mu, new_sigma, won) in updates.items():
            if fixed_baseline_checkpoint is not None and ckpt_id == fixed_baseline_checkpoint:
                continue
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

    # Isolate the role imbalance from each model's stored rating by injecting a
    # temporary virtual team member into the Mafia side. The bias is computed
    # dynamically from the current environment and the uncertainty of all players
    # in the match, so it does not permanently alter the stored model rating.
    all_ratings = mafia_ratings + village_ratings
    mafia_team = mafia_ratings + [_virtual_role_bias_rating(all_ratings, env)]
    village_team = village_ratings

    mafia_rank = 0 if winning_team == "Mafia" else 1
    village_rank = 1 if winning_team == "Mafia" else 0
    updated_mafia, updated_village = env.rate(
        [mafia_team, village_team], ranks=[mafia_rank, village_rank]
    )

    updated_mafia = updated_mafia[:-1]

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
