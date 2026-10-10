from __future__ import annotations
from transformers import AutoTokenizer
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xmlrpc.client import boolean

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
BRACKET_ACTION_RE = re.compile(r"\[\d+\]")
SFT_MODEL_ID = "fknuette/werwolf-sft"
SFT_ORIGIN_FILE = "sft_origin.json"


def is_sft_model(model: str, tokenizer=None) -> bool:
    """Identify the original SFT model or a merged checkpoint derived from it."""
    if model == SFT_MODEL_ID or getattr(tokenizer, "name_or_path", None) == SFT_MODEL_ID:
        return True
    origin_file = Path(model) / SFT_ORIGIN_FILE
    if origin_file.is_file():
        return json.loads(origin_file.read_text(encoding="utf-8")).get("base_model") == SFT_MODEL_ID
    return False


def sft_stop_token_ids(tokenizer, model: str) -> list[int] | None:
    """Stop the werwolf SFT checkpoint at the end of its assistant turn."""
    if not is_sft_model(model, tokenizer):
        return None

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    has_im_end = (
        isinstance(im_end_id, int)
        and im_end_id >= 0
        and tokenizer.convert_ids_to_tokens(im_end_id) == "<|im_end|>"
    )
    stop_token_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, im_end_id if has_im_end else None)
        if isinstance(token_id, int) and token_id >= 0
    }
    if not stop_token_ids:
        raise ValueError("Tokenizer does not provide an EOS or <|im_end|> token")
    return sorted(stop_token_ids)


def extract_phase(observation:str) -> Literal["Discuss", "Voting", "Action"]:
    # Findout in we will vote or not
    matches = re.findall(r'\[GAME\](.*)(?=\n|$)', observation)
    valid_matches = [m.strip() for m in matches if "invalid move" not in m.lower()]
    if not valid_matches:
        raise ValueError(f"No valid [GAME] markers found in observation: {observation!r}")
    phase_text = valid_matches[-1]
    if "Voting phase" in phase_text:
        return "Voting"
    elif "Discuss" in phase_text:
        return "Discuss"
    else:
        return "Action"

@dataclass
class ParsedResponse:
    raw_text: str
    reasoning: str
    action: str

# Here you have the possibility to enhance the observation prompt from the system
def build_agent_prompt(observation: str, phase: Literal["Discuss", "Voting", "Action"]) -> list:
    parts = observation.split("[GAME]")
    system_part = parts[1].strip()
    game_state = "[GAME]".join(parts[2:]).strip()
    if phase == "Voting":
        order = "You MUST vote. You MUST NOT discuss. You MUST NOT explain. You MUST NOT output anything except a valid bracketed number."
    elif phase == "Discuss":
        order = "You MUST discuss. Think privately. Output ONLY your public statement. Do NOT reveal hidden reasoning."
    else:
        order = "You MUST perform your role action. Output ONLY one valid bracketed number. Do NOT explain."
    
    prompt = f"Game State: {game_state}\n\nInstruction: {order}"

    messages = [
        {"role": "system", "content": system_part},
        {"role": "user", "content": prompt}
    ]
    return messages


def parse_model_response(raw_text: str) -> ParsedResponse:
    # reasoning_match = THINK_RE.search(raw_text)
    # action_match = ACTION_RE.search(raw_text)

    #reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    reasoning = ""

    stripped_text = raw_text.strip()
    bracket_match = BRACKET_ACTION_RE.search(raw_text)
    if bracket_match:
        action = bracket_match.group(0)
    elif stripped_text:
        action = stripped_text.splitlines()[0]
    else:
        action = ""

    return ParsedResponse(raw_text=raw_text, reasoning=reasoning, action=action)
