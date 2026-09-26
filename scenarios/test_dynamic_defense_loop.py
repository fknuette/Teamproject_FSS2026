"""CPU-only integration check for dynamic defense harvesting and rating."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from defense_harvest import is_defense_situation, judge_context


class DynamicDefenseTests(unittest.TestCase):
    def test_only_public_recent_accusations_are_harvested(self) -> None:
        observation = (
            "[GAME] Welcome. You are Player 2.\nYour role: Mafia\n"
            "[Player 1] Player 2 is suspicious; explain your vote.\n"
            "[GAME] Discuss phase: respond."
        )
        self.assertTrue(is_defense_situation(observation, 2))
        self.assertFalse(is_defense_situation(observation, 3))
        self.assertNotIn("Mafia", judge_context(observation, 2))
        self.assertIn("Player 2 is suspicious", judge_context(observation, 2))

    def test_self_play_to_scored_jsonl(self) -> None:
        fake_vllm = types.ModuleType("vllm")

        class SamplingParams:
            def __init__(self, **kwargs):
                self.n = kwargs.get("n", 1)

        class LLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, sampling):
                texts = [
                    "Ich habe meine Meinung geändert, weil Spieler 3 danach neue Informationen genannt hat.",
                    "Nein.",
                ]
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)
                                                       for text in texts[:sampling.n]])]

        fake_vllm.LLM = LLM
        fake_vllm.SamplingParams = SamplingParams

        fake_transformers = types.ModuleType("transformers")

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, model):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                return messages[1]["content"]

        fake_transformers.AutoTokenizer = AutoTokenizer
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

        first = "[GAME] Welcome. You are Player 1.\n[GAME] Discuss phase: begin."
        second = (
            "[GAME] Welcome. You are Player 2.\nYour role: Mafia\n"
            "[Player 1] Player 2 is suspicious. Explain your vote.\n"
            "[GAME] Discuss phase: respond."
        )

        class Environment:
            def __init__(self):
                self.turn = 0
                self.state = types.SimpleNamespace(game_state={"phase": "Discuss"})

            def reset(self, num_players):
                pass

            def get_observation(self):
                return [(1, first), (2, second)][self.turn]

            def step(self, action):
                self.turn += 1
                return self.turn == 2, {}

        fake_ta = types.ModuleType("textarena")
        fake_ta.make = lambda env_id: Environment()
        fake_agent_module = types.ModuleType("self_play_textarena")

        class Agent:
            def __init__(self, llm, tokenizer):
                pass

            def __call__(self, observation):
                return {"action": "public speech"}

        fake_agent_module.VLLMTextArenaAgent = Agent
        fake_trainer = types.ModuleType("grpo_training.cli")
        fake_trainer.run_training = lambda args: None

        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root / "src"))
        with patch.dict(sys.modules, {
            "vllm": fake_vllm, "transformers": fake_transformers,
            "torch": fake_torch, "textarena": fake_ta,
            "self_play_textarena": fake_agent_module, "grpo_training.cli": fake_trainer,
        }):
            from dynamic_defense_loop import harvest_complete_and_rate

            with tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                raw_dir, scenario_dir, completions_dir = (
                    base / "raw", base / "scenarios", base / "completions"
                )
                for directory in (raw_dir, scenario_dir, completions_dir):
                    directory.mkdir()
                args = argparse.Namespace(
                    tensor_parallel_size=1, gpu_memory_utilization=0.6,
                    num_players=3, games_per_iter=1, situations_per_game=2,
                    accusation_window=8, env_id="SecretMafia-v0",
                    num_completions=2, temperature=1.0, max_tokens=200,
                    judge_mode="mock", judge_model="",
                )
                files = harvest_complete_and_rate(
                    "fake-policy", "iter_1", raw_dir, scenario_dir, completions_dir, args
                )
                self.assertEqual(len(files), 1)
                records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 2)
                self.assertEqual({record["player_id"] for record in records}, {2})
                self.assertEqual({record["game_id"] for record in records}, {0})
                self.assertNotIn("Your role: Mafia", records[0]["judge_observation"])
                self.assertGreater(records[0]["reward"], records[1]["reward"])
                self.assertEqual(len(list(scenario_dir.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
