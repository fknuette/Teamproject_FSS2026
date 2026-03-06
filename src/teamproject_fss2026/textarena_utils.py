from __future__ import annotations

import re
from dataclasses import dataclass

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
BRACKET_ACTION_RE = re.compile(r"\[\d+\]")


@dataclass
class ParsedResponse:
    raw_text: str
    reasoning: str
    action: str


def build_agent_prompt(observation: str) -> str:
    return (
        "You are a competitive TextArena agent.\n"
        "Think step by step and put your reasoning into <think>...</think>.\n"
        "Then output exactly one final action in <action>...</action>.\n"
        "Do not add anything after </action>.\n\n"
        f"Observation:\n{observation}"
    )


def parse_model_response(raw_text: str) -> ParsedResponse:
    reasoning_match = THINK_RE.search(raw_text)
    action_match = ACTION_RE.search(raw_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    if action_match:
        action = action_match.group(1).strip()
    else:
        bracket_match = BRACKET_ACTION_RE.search(raw_text)
        action = bracket_match.group(0) if bracket_match else raw_text.strip().splitlines()[0]

    return ParsedResponse(raw_text=raw_text, reasoning=reasoning, action=action)
