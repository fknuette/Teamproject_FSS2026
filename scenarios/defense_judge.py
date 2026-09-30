"""Shared suspicion judge for fixed defense scenarios.

The default evaluator is a deterministic mock. ``mode=local`` uses only a
locally loaded vLLM model; no external API is involved.

Two APIs are provided:
- Legacy single-player API (``evaluate_suspicion``, ``build_judge_messages``):
  kept for backward compatibility with existing callers.
- Multi-player API (``evaluate_all_suspicions``, ``build_multi_judge_messages``):
  each judge call rates ALL other players in one response using numeric tags.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Protocol

# ---------------------------------------------------------------------------
# Legacy single-player API (kept for backward compatibility)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Multi-player API
# Each judge call returns numeric ratings for ALL players except themselves.
# Response format: <vote_N>NUMBER</vote_N> per player (1-10, no reasoning).
# ---------------------------------------------------------------------------

MULTI_VOTE_RE = re.compile(r"<\s*vote_(\d+)\s*>\s*(\d+)\s*<\s*/\s*vote_\d+\s*>", re.DOTALL)

_MULTI_JUDGE_RATING_INSTRUCTION = (
    "When asked to evaluate suspicion, output ONLY <vote_N>NUMBER</vote_N> tags, "
    "one per player. NUMBER is an integer from 1 (completely unsuspicious) to "
    "10 (undoubtedly guilty). No explanations, no other text."
)

_PLAYER_ID_RE = re.compile(r"You are Player\s+(\d+)", re.IGNORECASE)
_ROLE_BLOCK_RE = re.compile(
    r"(Your role\s*:.*?)(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL
)


def build_judge_system_prompt(observation: str, judge_id: int) -> str:
    """Build a villager system prompt for judge_id from the game observation's header.

    Mirrors the split done in ``build_agent_prompt`` (textarena_utils.py): the
    first ``[GAME]`` section contains general game info (welcome message, role,
    rules).  We replace the original player identity with the judge's ID and
    force the role to Villager/Village so no private role info leaks.
    """
    parts = observation.split("[GAME]")
    # parts[0] is text before the first [GAME] (usually empty)
    # parts[1] is the general game info section
    base = parts[1].strip() if len(parts) > 1 else ""

    # Replace "You are Player X" with the judge's actual player ID
    base = _PLAYER_ID_RE.sub(f"You are Player {judge_id}", base)

    # Replace any role/team block so Mafia judges are not revealed
    role_replacement = (
        "Your role: Villager\n"
        "Team: Village\n"
        "Description: A regular villager. Your goal is to identify all Mafia members."
    )
    base = _ROLE_BLOCK_RE.sub(role_replacement, base, count=1)

    return base.rstrip() + "\n\n" + _MULTI_JUDGE_RATING_INSTRUCTION


def parse_multi_ratings(raw_output: str, expected_player_ids: list[int]) -> dict[int, int]:
    """Parse all ``<vote_N>NUMBER</vote_N>`` tags.  Missing players fall back to 5."""
    result: dict[int, int] = {}
    for match in MULTI_VOTE_RE.finditer(raw_output):
        pid = int(match.group(1))
        score = max(1, min(10, int(match.group(2))))
        result[pid] = score
    for pid in expected_player_ids:
        if pid not in result:
            print(f"[WARNING] Judge missing rating for player {pid}, fallback to 5")
            result[pid] = 5
    return result


def build_multi_judge_messages(
    observation: str,
    public_game_state: str,
    response: str | None,
    judge_id: int,
    other_player_ids: list[int],
) -> list[dict[str, str]]:
    """Build messages for one multi-player judge call.

    The acting player stays in the scored set for the scenario, but the judge
    itself is never allowed to rate its own ID. The speaker is also excluded from
    the judge pool before selection, so the same player never acts as judge.
    """
    system = build_judge_system_prompt(observation, judge_id)
    other_player_ids = [pid for pid in other_player_ids if pid != judge_id]
    player_list = ", ".join(str(p) for p in other_player_ids)
    user = f"Game State:\n{public_game_state}\n\n"
    if response is not None:
        user += f"New public statement from the speaking player:\n{response}\n\n"
    user += (
        f"Rate the suspicion level for each of the following players: [{player_list}]. "
        "Output ONLY <vote_N>NUMBER</vote_N> tags, one per player. "
        "Example: <vote_2>7</vote_2>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def evaluate_all_suspicions(
    observation: str,
    public_game_state: str,
    response: str | None,
    judge_ids: list[int],
    all_player_ids: list[int],
    evaluator: "Evaluator",
) -> dict[int, float]:
    """Rate suspicion of every player using ``len(judge_ids)`` LLM calls.

    Each judge rates all players except themselves in one call.  Returns a dict
    mapping player_id → mean suspicion across all judges that rated that player.
    """
    ratings: dict[int, list[int]] = {p: [] for p in all_player_ids}
    for judge_id in judge_ids:
        other_ids = [p for p in all_player_ids if p != judge_id]
        msgs = build_multi_judge_messages(observation, public_game_state, response, judge_id, other_ids)
        raw = evaluator(msgs, judge_id)
        per_player = parse_multi_ratings(raw, other_ids)
        for pid, score in per_player.items():
            ratings[pid].append(score)
    return {p: mean(r) if r else 5.0 for p, r in ratings.items()}


# ---------------------------------------------------------------------------
# Evaluator implementations (shared by both APIs)
# ---------------------------------------------------------------------------

class Evaluator(Protocol):
    def __call__(self, messages: list[dict[str, str]], villager_id: int) -> str: ...


def mock_local_llm_evaluator(messages: list[dict[str, str]], villager_id: int) -> str:
    """Deterministic stand-in for smoke tests.

    Detects which API is being used from the user message content:
    - Multi-player (new): outputs ``<vote_N>NUMBER</vote_N>`` per player.
    - Single-player (legacy): outputs ``<thought>...</thought><vote>LETTER</vote>``.
    """
    user_text = messages[1]["content"]

    # --- Multi-player path ---
    if "Rate the suspicion level for each of the following players:" in user_text:
        # Extract player IDs from the prompt: "players: [0, 2, 3, ...]"
        id_match = re.search(r"\[([0-9,\s]+)\]", user_text)
        other_ids: list[int] = []
        if id_match:
            other_ids = [int(x.strip()) for x in id_match.group(1).split(",") if x.strip().isdigit()]

        # Detect presence of a new statement to compute length-based signal
        has_response = "New public statement from the speaking player:" in user_text
        if has_response:
            response_text = user_text.split("New public statement from the speaking player:\n", 1)[1]
            response_text = response_text.split("\n\nRate the suspicion", 1)[0]
            change = 1 if len(response_text.strip()) >= 25 else 0
        else:
            change = 0

        base = 5
        tags = "".join(f"<vote_{pid}>{max(1, min(10, base + change))}</vote_{pid}>" for pid in other_ids)
        return tags

    # --- Legacy single-player path ---
    if "Neue öffentliche Verteidigungsrede:" not in user_text:
        rating = (7, 8, 7)[(villager_id - 1) % 3]
        reason = "Die Vorwürfe sind noch unbeantwortet."
    else:
        response = user_text.split("Neue öffentliche Verteidigungsrede:\n", 1)[1].split(
            "\n\nBewerte den Verdacht NACH", 1
        )[0]
        change = 2 if len(response.strip()) >= 80 else 1 if len(response.strip()) >= 25 else 0
        rating = max(1, (7, 8, 7)[(villager_id - 1) % 3] - change)
        reason = "Die Antwort geht auf den Verdacht ein." if change else "Die Antwort bleibt knapp."
    letter = "ABCDEFGHIJ"[rating - 1]
    return f"<thought>{reason}</thought>\n<vote>{letter}</vote>"


class LocalVLLMEvaluator:
    """One locally hosted vLLM instance reused for all judge calls."""

    def __init__(self, model: str, gpu_memory_utilization: float = 0.6):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=256)

    def __call__(self, messages: list[dict[str, str]], villager_id: int) -> str:
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
    """Legacy single-player suspicion evaluation (kept for backward compatibility)."""
    ratings = []
    for villager_id in range(1, 4):
        messages = build_judge_messages(observation, response, villager_id)
        raw_output = evaluator(messages, villager_id)
        ratings.append(parse_rating(raw_output))
    return mean(ratings)
