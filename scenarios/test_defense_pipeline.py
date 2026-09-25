"""CPU-only contract checks for the defense scenario data flow."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from assign_defense_rewards import assign_defense_rewards
from defense_judge import mock_local_llm_evaluator, parse_rating
from generate_suspicion_observations import generate_observations


class DefensePipelineTests(unittest.TestCase):
    def test_parser_accepts_whitespace_and_warns_on_fallback(self) -> None:
        self.assertEqual(parse_rating("<thought>kurz</thought>\n<vote>\n h \n</vote>"), 8)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(parse_rating("<vote>K</vote>"), 5)
        self.assertIn("[WARNING] Fallback genutzt!", output.getvalue())

    def test_json_observation_to_completion_to_reward(self) -> None:
        # Stub only optional GPU packages; exercise the real prompt and JSONL path.
        fake_vllm = types.ModuleType("vllm")

        class SamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm.SamplingParams = SamplingParams
        fake_vllm.LLM = object
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = object
        project_root = Path(__file__).resolve().parents[1]
        with patch.dict(sys.modules, {"vllm": fake_vllm, "transformers": fake_transformers}):
            sys.path.insert(0, str(project_root / "src"))
            from completion_generator import generate_completions

            class Tokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return messages[1]["content"]

            class LLM:
                def generate(self, prompts, sampling):
                    texts = ["Ich erkläre den Wechsel: Neue Aussage von Spieler 3 änderte meine Einschätzung.", "Nein."]
                    return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t) for t in texts])]

            with tempfile.TemporaryDirectory() as temp_dir:
                scenarios = generate_observations(1, Path(temp_dir) / "observations", mock_local_llm_evaluator)
                output = Path(temp_dir) / "completions.jsonl"
                records = generate_completions(
                    str(scenarios[0]), "unused", num_completions=2, output=str(output),
                    llm=LLM(), tokenizer=Tokenizer(),
                )
                original = json.loads(scenarios[0].read_text(encoding="utf-8"))
                self.assertEqual(records[0]["observation"], original["observation"])
                self.assertEqual(records[0]["verdacht_pre"], original["verdacht_pre"])
                scored = assign_defense_rewards(str(output), evaluator=mock_local_llm_evaluator)
                self.assertEqual(len(scored), 2)
                self.assertEqual(scored[0]["reward"], scored[0]["verdacht_pre"] - scored[0]["verdacht_post"])
                self.assertGreater(scored[0]["reward"], scored[1]["reward"])
                self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
