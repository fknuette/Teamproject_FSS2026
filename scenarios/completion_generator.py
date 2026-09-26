"""
Generate N completions for ONE fixed scenario (raw .txt or JSON observation) and save them in
the JSONL format that your GRPODataset can read directly.

Can be used two ways:
  1) From the command line (thin CLI wrapper around the function).
  2) Imported from another script:
         from generate_completions import generate_completions
         records = generate_completions(scenario="prompts/s.txt", model="...", ...)

Design decisions (intentional, see comments):
- We save the RAW observation (from the .txt or JSON 'observation'), NOT the already-built prompt.
  data.py rebuilds the prompt itself from the observation (extract_phase +
  build_agent_prompt + apply_chat_template). Saving the finished prompt would run
  it through the chain twice.
- For SAMPLING, however, we build the prompt with exactly the same chain as the
  real self_play, so the completions are drawn from the same distribution.
- All N completions get the same game_id -> data.py treats them as ONE GRPO group
  (advantage normalization within the scenario).
- reward is written as a placeholder 0.0; the evaluator step (colleague) fills /
  overwrites it later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# --- add src to path so textarena_utils can be found (as in your other scripts) ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from teamproject_fss2026.textarena_utils import build_agent_prompt, extract_phase


def generate_completions(
    scenario: str,
    model: str,
    player_id: int = 0,
    num_completions: int = 8,
    game_id: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 200,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.6,
    output: str = "",
    llm: Optional["LLM"] = None,
    tokenizer=None,
) -> list[dict]:
    """
    Generate `num_completions` completions for the raw observation in `scenario`
    and write them as GRPODataset-compatible JSONL.

    Returns the list of written records (so a calling script can use them directly).

    If `llm` and `tokenizer` are passed in, they are reused instead of loading a
    fresh model -- useful when a caller generates completions for MANY scenarios
    and wants to load the model only once.
    """
    # --- load raw observation ---
    scenario_path = Path(scenario)
    observation = scenario_path.read_text(encoding="utf-8")
    scenario_metadata: dict = {}
    if scenario_path.suffix.lower() == ".json":
        scenario_data = json.loads(observation)
        if not isinstance(scenario_data, dict) or not isinstance(scenario_data.get("observation"), str):
            raise ValueError(f"JSON scenario needs a string 'observation': {scenario_path}")
        observation = scenario_data["observation"]
        for key in ("verdacht_pre", "judge_observation", "judge_mode", "judge_model"):
            if key in scenario_data:
                scenario_metadata[key] = scenario_data[key]
    if not observation.strip():
        raise ValueError(f"Scenario file is empty: {scenario_path}")

    # --- target path ---
    if output:
        out_path = Path(output)
    else:
        out_dir = Path(__file__).resolve().parent / "completions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scenario_{game_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load model/tokenizer only if not provided by the caller ---
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model)
    if llm is None:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # --- build the prompt with the REAL chain (identical to self_play) ---
    phase = extract_phase(observation)
    own_prompt = build_agent_prompt(observation, phase)
    prompt_text = tokenizer.apply_chat_template(
        own_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("=" * 70)
    print(f"Scenario     : {scenario_path}")
    print(f"Phase        : {phase}")
    print(f"Model        : {model}")
    print(f"Completions  : {num_completions} (game_id={game_id})")
    print(f"Output       : {out_path}")
    print("=" * 70)

    # --- N completions in ONE vLLM call (efficient: continuous batching) ---
    sampling = SamplingParams(
        n=num_completions,               # n completions for the same prompt
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([prompt_text], sampling)
    completions = [o.text for o in outputs[0].outputs]

    # --- build records in GRPODataset-compatible format ---
    # Fields exactly as TurnRecord / data.py expects:
    # game_id, observation (RAW!), response, reward, player_id, turn_id
    records: list[dict] = []
    for i, completion in enumerate(completions):
        records.append({
            "game_id": game_id,          # ALL equal -> one GRPO group
            "observation": observation,  # RAW observation, NOT the prompt
            "response": completion,       # generated text
            "reward": 0.0,                # placeholder -> evaluator fills this
            "player_id": player_id,
            "turn_id": i,                 # only to distinguish within the group
            **scenario_metadata,
        })

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Saved] {len(records)} completions -> {out_path}")
    print("\n--- Preview ---")
    for i, c in enumerate(completions):
        preview = c.strip().replace("\n", " ")
        print(f"[{i}] {preview[:120]}{'...' if len(preview) > 120 else ''}")

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate N completions for one fixed scenario")
    p.add_argument("--scenario", type=str, required=True,
                   help="Path to a raw .txt observation or a .json scenario with 'observation'.")
    p.add_argument("--model", type=str, required=True,
                   help="Model path or HF name used for sampling.")
    p.add_argument("--player-id", type=int, default=0,
                   help="player_id written into the training format (metadata only).")
    p.add_argument("--num-completions", type=int, default=8,
                   help="Number of completions (= size of the GRPO group). Default: 8")
    p.add_argument("--game-id", type=int, default=0,
                   help="game_id for ALL completions of this scenario (one group).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (>0, so the completions actually vary!).")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Max tokens per completion (Discuss default from self_play: 200).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--output", type=str, default="",
                   help="Target JSONL. Empty -> completions/scenario_<game_id>.jsonl next to the script.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_completions(
        scenario=args.scenario,
        model=args.model,
        player_id=args.player_id,
        num_completions=args.num_completions,
        game_id=args.game_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output=args.output,
    )


if __name__ == "__main__":
    main()
