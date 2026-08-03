"""Persistent registry of checkpoints and their per-role TrueSkill ratings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import trueskill

ROLES: tuple[str, ...] = ("Mafia", "Doctor", "Detective", "Villager")

# TrueSkill environment defaults
_DEFAULT_MU = 25.0
_DEFAULT_SIGMA = 25.0 / 3
_DEFAULT_BETA = 25.0 / 6
_DEFAULT_TAU = 25.0 / 300
_DEFAULT_DRAW_PROB = 0.0


@dataclass
class RoleRating:
    """TrueSkill rating for a single (checkpoint, role) pair."""

    mu: float = _DEFAULT_MU
    sigma: float = _DEFAULT_SIGMA
    games: int = 0
    wins: int = 0


@dataclass
class CheckpointEntry:
    """All metadata and ratings for one checkpoint."""

    checkpoint_id: str
    path: str
    ratings: dict[str, RoleRating] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for role in ROLES:
            self.ratings.setdefault(role, RoleRating())


class CheckpointRegistry:
    """JSON-backed persistent store of checkpoints and their TrueSkill ratings.

    All mutating methods update only the in-memory state; call ``save()`` to
    persist changes atomically (write-then-rename).
    """

    DEFAULT_REGISTRY_FILENAME = "checkpoint_registry.json"

    def __init__(self, registry_path: Path) -> None:
        self._path = Path(registry_path)
        self._checkpoints: dict[str, CheckpointEntry] = {}
        self._env_params: dict[str, float] = {
            "mu": _DEFAULT_MU,
            "sigma": _DEFAULT_SIGMA,
            "beta": _DEFAULT_BETA,
            "tau": _DEFAULT_TAU,
            "draw_probability": _DEFAULT_DRAW_PROB,
        }
        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        checkpoint_id: str,
        path: str,
        *,
        overwrite: bool = False,
    ) -> CheckpointEntry:
        """Add a checkpoint to the registry.

        If the checkpoint already exists and ``overwrite=False`` (default),
        the existing entry is returned unchanged.
        """
        if checkpoint_id in self._checkpoints and not overwrite:
            return self._checkpoints[checkpoint_id]
        entry = CheckpointEntry(checkpoint_id=checkpoint_id, path=path)
        self._checkpoints[checkpoint_id] = entry
        return entry

    def get(self, checkpoint_id: str) -> CheckpointEntry:
        """Return the entry for *checkpoint_id*, raising ``KeyError`` if absent."""
        return self._checkpoints[checkpoint_id]

    def all_ids(self) -> list[str]:
        """Return all registered checkpoint IDs."""
        return list(self._checkpoints.keys())

    def update_rating(
        self,
        checkpoint_id: str,
        role: str,
        new_mu: float,
        new_sigma: float,
        won: bool,
    ) -> None:
        """Update the in-memory rating for *(checkpoint_id, role)*.

        Does **not** persist to disk; call ``save()`` when ready.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}. Expected one of {ROLES}")
        rating = self._checkpoints[checkpoint_id].ratings[role]
        rating.mu = new_mu
        rating.sigma = new_sigma
        rating.games += 1
        if won:
            rating.wins += 1

    def get_trueskill_rating(self, checkpoint_id: str, role: str) -> trueskill.Rating:
        """Return a ``trueskill.Rating`` for *(checkpoint_id, role)*."""
        r = self._checkpoints[checkpoint_id].ratings[role]
        env = self._build_trueskill_env()
        return env.create_rating(mu=r.mu, sigma=r.sigma)

    def save(self) -> None:
        """Atomically persist the registry to disk (write .tmp, then rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        data = {
            "checkpoints": {
                ckpt_id: {
                    "path": entry.path,
                    **{
                        role: asdict(entry.ratings[role])
                        for role in ROLES
                    },
                }
                for ckpt_id, entry in self._checkpoints.items()
            },
            "trueskill_env": self._env_params,
        }
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._env_params = raw.get("trueskill_env", self._env_params)
        for ckpt_id, ckpt_data in raw.get("checkpoints", {}).items():
            ratings: dict[str, RoleRating] = {}
            for role in ROLES:
                if role in ckpt_data:
                    rd = ckpt_data[role]
                    ratings[role] = RoleRating(
                        mu=rd["mu"],
                        sigma=rd["sigma"],
                        games=rd["games"],
                        wins=rd["wins"],
                    )
                else:
                    ratings[role] = RoleRating()
            self._checkpoints[ckpt_id] = CheckpointEntry(
                checkpoint_id=ckpt_id,
                path=ckpt_data["path"],
                ratings=ratings,
            )

    def _build_trueskill_env(self) -> trueskill.TrueSkill:
        return trueskill.TrueSkill(
            mu=self._env_params["mu"],
            sigma=self._env_params["sigma"],
            beta=self._env_params["beta"],
            tau=self._env_params["tau"],
            draw_probability=self._env_params["draw_probability"],
        )
