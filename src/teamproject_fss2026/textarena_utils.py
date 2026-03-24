from __future__ import annotations
from transformers import AutoTokenizer
import re
from dataclasses import dataclass
from xmlrpc.client import boolean

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
BRACKET_ACTION_RE = re.compile(r"\[\d+\]")


@dataclass
class ParsedResponse:
    raw_text: str
    reasoning: str
    action: str

# Here you have the possibility to enhance the observation prompt from the system
def build_agent_prompt(observation: str) -> str:
    parts = observation.split("[GAME]")
    system_part = parts[1].strip()
    game_state = "[GAME]".join(parts[2:]).strip()
    matches = re.findall(r'\[GAME\](.*)(?=\n|$)', observation)
    order = matches[-1].strip()
    if "Voting phase" in order:
        order = "You MUST vote. You MUST NOT discuss. You MUST NOT explain. You MUST NOT output anything except a valid bracketed number."
    elif "Discuss" in order:
        order = "You MUST discuss. Think privately. Output ONLY your public statement. Do NOT reveal hidden reasoning."
    else:
        order = "You MUST perform your role action. Output ONLY one valid bracketed number. Do NOT explain."
    
    prompt = f"Game State: {game_state}\n\nInstruction: {order}"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    #prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": system_part},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    #model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return text


def parse_model_response(raw_text: str) -> ParsedResponse:
    # reasoning_match = THINK_RE.search(raw_text)
    # action_match = ACTION_RE.search(raw_text)

    #reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    reasoning = ""
    
    bracket_match = BRACKET_ACTION_RE.search(raw_text)
    action = bracket_match.group(0) if bracket_match else raw_text.strip().splitlines()[0]

    return ParsedResponse(raw_text=raw_text, reasoning=reasoning, action=action)
