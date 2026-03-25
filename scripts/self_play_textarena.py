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
import re
from textarena.core import Agent
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from teamproject_fss2026.textarena_utils import build_agent_prompt, parse_model_response


@dataclass
class TurnRecord:
    game_id: int
    observation: str
    response: str
    reward: float = 0.0
    player_id: int = 0
    turn_id: int = 0


class VLLMTextArenaAgent(Agent):
    def __init__(self, llm: LLM, tokinizer: AutoTokenizer) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokinizer

    def __call__(self, observation: str) -> dict[str, str]:
        
        # Findout in we will vote or not
        matches = re.findall(r'\[GAME\](.*)(?=\n|$)', observation)
        valid_matches = [m.strip() for m in matches if "invalid move" not in m.lower()]
        phase_text = valid_matches[-1]
        if "Voting phase" in phase_text:
            phase = "Voting"
        elif "Discuss" in phase_text:
            phase = "Discuss"
        else:
            phase = "Action"

        # Tag-Modus: Darf reden, stoppt nur am nächsten Block-Trenner
        if phase == "Discuss":
            current_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=200,
                # stop=["###"] # Stoppt erst am nächsten Block
            )
        else:
            # Logik-Check: Nacht (Nur Nummer) vs. Tag (Reden/Rechnen)
            current_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=100,
                # stop=[".", "\n", "]", " "] # Stoppt sofort nach der Zahl/Klammer
            )
            # Hier evtl. temperature > 0 lassen für natürlichere Sprache
        
        
        own_prompt = build_agent_prompt(observation, phase)
        
        
        prompt = self.tokenizer.apply_chat_template(
                    own_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
        
        outputs = self.llm.generate([prompt], current_params)
        raw_text = outputs[0].outputs[0].text
        parsed = parse_model_response(raw_text)
        return {
            "response": raw_text,
            "action": parsed.action,
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

    llm = LLM(model=args.model_a, tensor_parallel_size=args.tensor_parallel_size,)
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)

    all_records: list[TurnRecord] = []

    for game_id in range(args.num_games):
        env = ta.make(args.env_id)
        env.reset(num_players=6)

        agents: dict[int, VLLMTextArenaAgent] = {
            0: VLLMTextArenaAgent(llm, tokenizer),
            1: VLLMTextArenaAgent(llm, tokenizer),
            2: VLLMTextArenaAgent(llm, tokenizer),
            3: VLLMTextArenaAgent(llm, tokenizer),
            4: VLLMTextArenaAgent(llm, tokenizer),
            5: VLLMTextArenaAgent(llm, tokenizer),
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
                    observation=observation,
                    response=agent_out["response"],
                )
            )
            turn_id += 1

        rewards, _game_info = env.close()
        reward_map = _normalize_rewards(rewards)

        for record in game_records:
            record.reward = reward_map.get(record.player_id, 0.0)

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
