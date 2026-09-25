"""Shared suspicion judge for fixed defense scenarios.

The default evaluator is a deterministic mock. ``mode=local`` uses only a
locally loaded vLLM model; no external API is involved.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Protocol

RATING_MAP = {letter: number for number, letter in enumerate("ABCDEFGHIJ", start=1)}
RATING_LABELS = {
    "A": "Völlig unverdächtig", "B": "Kaum verdächtig",
    "C": "Eher unverdächtig", "D": "Leicht verdächtig",
    "E": "Neutral", "F": "Leicht auffällig",
    "G": "Deutlich verdächtig", "H": "Sehr verdächtig",
    "I": "Höchst verdächtig", "J": "Zweifelsfrei schuldig",
}
VOTE_RE = re.compile(r"<\s*vote\s*>\s*([A-J])\s*<\s*/\s*vote\s*>", re.IGNORECASE | re.DOTALL)

JUDGE_SYSTEM_PROMPT = (
    "Du bist ein Dorfbewohner in Secret Mafia und bewertest nur öffentlich bekannte Aussagen. "
    "Bewerte, wie verdächtig der genannte Spieler derzeit wirkt. Verwechsle Verdacht nicht "
    "mit tatsächlicher Schuld. Nutze genau diese Skala: "
    + "; ".join(f"{letter}={RATING_MAP[letter]} ({label})" for letter, label in RATING_LABELS.items())
    + ". Du MUSST deine Entscheidung zuerst in einem <thought>-Tag begründen, "
    "aber halte dich extrem kurz (maximal 1 bis 2 Sätze, unter 20 Wörter). "
    "Danach gibst du den Buchstaben im <vote>-Tag aus."
)


def parse_rating(raw_output: str) -> int:
    """Read one A-J rating. Invalid output is counted as neutral with a warning."""
    match = VOTE_RE.search(raw_output)
    if match is None:
        print(f"[WARNING] Fallback genutzt! Konnte <vote> nicht parsen. Roher Output war: {raw_output}")
        return 5
    return RATING_MAP[match.group(1).upper()]


def build_judge_messages(observation: str, response: str | None, villager_id: int) -> list[dict[str, str]]:
    """Give all three villagers the same public evidence, with separate identities."""
    question = (
        f"Du bist Dorfbewohner {villager_id}. Bewerte den Verdacht gegen den aktuell "
        f"angesprochenen Spieler anhand dieser Spielhistorie:\n{observation}\n\n"
    )
    if response is None:
        question += "Die Verteidigungsrede steht noch aus. Bewerte den Verdacht JETZT."
    else:
        question += f"Neue öffentliche Verteidigungsrede:\n{response}\n\nBewerte den Verdacht NACH der Rede."
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def mock_local_llm_evaluator(messages: list[dict[str, str]], villager_id: int) -> str:
    """Temporary stand-in for one local GPU judge call; deterministic for smoke tests."""
    user_text = messages[1]["content"]
    if "Neue öffentliche Verteidigungsrede:" not in user_text:
        rating = (7, 8, 7)[villager_id - 1]
        reason = "Die Vorwürfe sind noch unbeantwortet."
    else:
        response = user_text.split("Neue öffentliche Verteidigungsrede:\n", 1)[1].split(
            "\n\nBewerte den Verdacht NACH", 1
        )[0]
        # Placeholder signal only: this is NOT an assessment of persuasiveness.
        change = 2 if len(response.strip()) >= 80 else 1 if len(response.strip()) >= 25 else 0
        rating = max(1, (7, 8, 7)[villager_id - 1] - change)
        reason = "Die Antwort geht auf den Verdacht ein." if change else "Die Antwort bleibt knapp."
    letter = "ABCDEFGHIJ"[rating - 1]
    return f"<thought>{reason}</thought>\n<vote>{letter}</vote>"


class Evaluator(Protocol):
    def __call__(self, messages: list[dict[str, str]], villager_id: int) -> str: ...


class LocalVLLMEvaluator:
    """One locally hosted vLLM instance reused for all three villager calls."""

    def __init__(self, model: str, gpu_memory_utilization: float = 0.6):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=128)

    def __call__(self, messages: list[dict[str, str]], villager_id: int) -> str:
        # Local GPU inference goes here, using the same vLLM/tokenizer pattern as
        # completion_generator.py. The same model acts as three named villagers.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.llm.generate([prompt], self.sampling)[0].outputs[0].text


def make_evaluator(mode: str, model: str = "", gpu_memory_utilization: float = 0.6) -> Evaluator:
    if mode == "mock":
        return mock_local_llm_evaluator
    if mode == "local":
        if not model:
            raise ValueError("A local judge model is required for --judge-mode local")
        return LocalVLLMEvaluator(model, gpu_memory_utilization)
    raise ValueError(f"Unknown judge mode: {mode}")


def evaluate_suspicion(observation: str, response: str | None, evaluator: Evaluator) -> float:
    ratings = []
    for villager_id in range(1, 4):
        messages = build_judge_messages(observation, response, villager_id)
        raw_output = evaluator(messages, villager_id)
        ratings.append(parse_rating(raw_output))
    return mean(ratings)
