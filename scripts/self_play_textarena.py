from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import textarena as ta
from textarena.core import Agent
from vllm import LLM, SamplingParams

from teamproject_fss2026.textarena_utils import build_agent_prompt, parse_model_response


@dataclass
class TurnRecord:
    game_id: int
    turn_id: int
    player_id: int
    model_name: str
    observation: str
    prompt: str
    raw_response: str
    reasoning_trace: str
    action: str
    raw_env_reward: float = 0.0
    final_reward: float = 0.0
    won: int = 0


class VLLMTextArenaAgent(Agent):
    def __init__(self, llm: LLM, sampling_params: SamplingParams, model_name: str):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.model_name = model_name

    def __call__(self, observation: str) -> dict[str, str]:
        prompt = build_agent_prompt(observation)
        outputs = self.llm.generate([prompt], self.sampling_params)
        raw_text = outputs[0].outputs[0].text
        parsed = parse_model_response(raw_text)
        return {
            "prompt": prompt,
            "raw_response": parsed.raw_text,
            "reasoning_trace": parsed.reasoning,
            "action": parsed.action,
            "model_name": self.model_name,
        }


def _normalize_rewards(rewards: Any) -> dict[int, float]:
    if isinstance(rewards, dict):
        return {int(k): float(v) for k, v in rewards.items()}
    if isinstance(rewards, (list, tuple)):
        return {i: float(v) for i, v in enumerate(rewards)}
    raise ValueError(f"Unsupported rewards format: {type(rewards)}")


def run_self_play(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llm_a = LLM(model=args.model_a, tensor_parallel_size=args.tensor_parallel_size,)
    # llm_b = LLM(model=args.model_b, tensor_parallel_size=args.tensor_parallel_size,)

    import ipdb; ipdb.set_trace()  # Debug breakpoint to inspect arguments and flow
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    all_records: list[TurnRecord] = []

    for game_id in range(args.num_games):
        env = ta.make(args.env_id)
        env.reset(num_players=6)

        agents: dict[int, VLLMTextArenaAgent] = {
            0: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
            1: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
            2: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
            3: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
            4: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
            5: VLLMTextArenaAgent(llm_a, sampling_params, args.model_a),
        }

        game_records: list[TurnRecord] = []
        done = False
        turn_id = 0

        while not done:
            player_id, observation = env.get_observation()
            agent_out = agents[player_id](observation)
            done, _ = env.step(action=agent_out["action"])

            game_records.append(
                TurnRecord(
                    game_id=game_id,
                    turn_id=turn_id,
                    player_id=player_id,
                    model_name=agent_out["model_name"],
                    observation=observation,
                    prompt=agent_out["prompt"],
                    raw_response=agent_out["raw_response"],
                    reasoning_trace=agent_out["reasoning_trace"],
                    action=agent_out["action"],
                )
            )
            turn_id += 1

        rewards, _game_info = env.close()
        reward_map = _normalize_rewards(rewards)
        max_reward = max(reward_map.values())

        for record in game_records:
            r = reward_map.get(record.player_id, 0.0)
            record.raw_env_reward = r
            record.won = int(r == max_reward)
            record.final_reward = float(record.won)

        all_records.extend(game_records)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} turn records to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect TextArena self-play traces with vLLM.")
    parser.add_argument("--env-id", type=str, default="TicTacToe-v0")
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/selfplay_traces.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_self_play(args)


if __name__ == "__main__":
    main()
