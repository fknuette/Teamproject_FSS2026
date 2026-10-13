"""On-demand LoRA chain-merge for TrueSkill evaluation.

Each iter_X LoRA adapter is a delta from the *previous* merged model, not from
the baseline directly.  To reconstruct iter_X without its merged_model dir we
must apply adapters in order: baseline → iter_1 → iter_2 → … → iter_X.

Public API
----------
build_merged_models_for_window(checkpoint_dir, baseline_model, window_ids, all_discovered)
    Creates merged_model dirs for window checkpoints that are missing them.
    Returns the set of checkpoint IDs for which a merged_model was newly created.

cleanup_created_merged_models(checkpoint_dir, created_ids, keep_window)
    Deletes merged_model dirs that were created on-demand and are no longer needed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo_training.models import merge_lora_adapter


LORA_ADAPTER_SUBPATH = Path("lora_adapter") / "final"
MERGED_MODEL_SUBDIR = "merged_model"


def _iter_sort_key(ckpt_id: str) -> tuple[int, str]:
    _, _, suffix = ckpt_id.partition("_")
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, ckpt_id)


def _is_iter_checkpoint(ckpt_id: str) -> bool:
    return ckpt_id.startswith("iter_")


def build_merged_models_for_window(
    checkpoint_dir: Path,
    baseline_model: str,
    window_checkpoint_ids: set[str],
    all_discovered: list[tuple[str, str]],
) -> set[str]:
    """Build missing merged_model dirs for all iter_* checkpoints in *window_checkpoint_ids*.

    Parameters
    ----------
    checkpoint_dir:
        Root directory containing iter_* subdirectories.
    baseline_model:
        HuggingFace model ID or path used as the chain's starting point.
    window_checkpoint_ids:
        The set of checkpoint IDs that will participate in this eval round.
        Non-iter IDs (e.g. baseline HF model IDs) are silently skipped.
    all_discovered:
        Sorted list of (checkpoint_id, model_path) pairs from
        eval_main._discover_checkpoints().  Used to determine the full ordered
        sequence of iterations.

    Returns
    -------
    set[str]
        IDs of checkpoints for which a merged_model directory was newly created
        by this call.  Does NOT include checkpoints that already had one.
    """
    all_iter_ids = [cid for cid, _ in all_discovered if _is_iter_checkpoint(cid)]
    window_iter_ids = sorted(
        [cid for cid in window_checkpoint_ids if _is_iter_checkpoint(cid)],
        key=_iter_sort_key,
    )

    if not window_iter_ids:
        return set()

    # Determine which iter checkpoints need a merged_model built.
    missing = {
        cid for cid in window_iter_ids
        if not (checkpoint_dir / cid / MERGED_MODEL_SUBDIR).is_dir()
    }
    if not missing:
        return set()

    # To build iter_X we may need to walk back through predecessors.
    # Find the earliest missing checkpoint and work forward from the nearest
    # anchor (an iter that already has a merged_model, or the baseline).
    earliest_missing_idx = min(
        all_iter_ids.index(cid) for cid in missing if cid in all_iter_ids
    )

    # Find anchor: the nearest predecessor with an existing merged_model.
    anchor_base: str = baseline_model
    for cid in reversed(all_iter_ids[:earliest_missing_idx]):
        merged_dir = checkpoint_dir / cid / MERGED_MODEL_SUBDIR
        if merged_dir.is_dir():
            anchor_base = str(merged_dir)
            earliest_missing_idx = all_iter_ids.index(cid) + 1
            break

    # Walk forward from the anchor, building merged models where needed.
    created: set[str] = set()
    current_base = anchor_base

    # We need to walk every iter from the anchor up to the last missing one
    # in the window (intermediate merges are needed even if not in window).
    last_missing_idx = max(
        all_iter_ids.index(cid) for cid in missing if cid in all_iter_ids
    )

    for cid in all_iter_ids[earliest_missing_idx: last_missing_idx + 1]:
        merged_dir = checkpoint_dir / cid / MERGED_MODEL_SUBDIR
        lora_dir = checkpoint_dir / cid / LORA_ADAPTER_SUBPATH

        if merged_dir.is_dir():
            # Already exists — just advance the base pointer.
            current_base = str(merged_dir)
            continue

        if not lora_dir.is_dir():
            print(
                f"[LoRAChainMerge] Warning: {cid} has no lora_adapter/final — skipping."
            )
            current_base = str(merged_dir)  # optimistic; won't be dir, next iter will also skip
            continue

        print(f"[LoRAChainMerge] Building {cid}/merged_model from base={Path(current_base).name} ...")
        merge_lora_adapter(
            base_model=current_base,
            adapter_dir=str(lora_dir),
            merged_output_dir=str(merged_dir),
        )
        created.add(cid)
        current_base = str(merged_dir)
        print(f"[LoRAChainMerge] {cid}/merged_model ready.")

    return created


def cleanup_created_merged_models(
    checkpoint_dir: Path,
    created_ids: set[str],
    keep_window: set[str],
) -> None:
    """Delete on-demand-created merged_model dirs that are no longer needed.

    Parameters
    ----------
    checkpoint_dir:
        Root directory containing iter_* subdirectories.
    created_ids:
        Checkpoint IDs whose merged_model was created by this session
        (returned by build_merged_models_for_window).
    keep_window:
        Checkpoint IDs that should be retained (the upcoming eval window).
        Everything in created_ids that is NOT in keep_window is deleted.
    """
    to_delete = created_ids - keep_window
    for cid in sorted(to_delete, key=_iter_sort_key):
        merged_dir = checkpoint_dir / cid / MERGED_MODEL_SUBDIR
        if merged_dir.is_dir():
            try:
                shutil.rmtree(merged_dir)
                print(f"[LoRAChainMerge] Deleted on-demand merged_model: {merged_dir}")
            except Exception as exc:
                print(f"[LoRAChainMerge] Could not delete {merged_dir}: {exc}")
