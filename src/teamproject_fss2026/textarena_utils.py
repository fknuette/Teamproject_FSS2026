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

# Here you have the possibility to enhance the observation prompt from the system
def build_agent_prompt(observation: str) -> str:
    
    final_prompt = ""
    
    # Instruktion
    final_prompt += "### Instruction:\n"
    final_prompt += "You are a player in a text-based social deduction game. Your goal is to win by outsmarting other players through strategic thinking and deception. Carefully analyze the information provided and make your move accordingly.\n\n"
    
    # Context
    final_prompt += "### Context:\n"
    final_prompt += observation + "\n\n"
    
    # Task
    final_prompt += "### Task:\n"
    final_prompt += "If the phase is NIGHT:\n"
    final_prompt += "- choose exactly one valid target. This means after ANSWER you put just ONE Number!\n\n"
    final_prompt += "If the phase is DAY:\n"
    final_prompt += "- speak naturally with other players. This means you do after ANSWER you do things what you want to say to the others!\n\n"
    # Rules
    # Answer
    final_prompt += "### Answer:\n"
    final_prompt += "I will vote for player ["
    return final_prompt
    


def parse_model_response(raw_text: str) -> ParsedResponse:
    # reasoning_match = THINK_RE.search(raw_text)
    # action_match = ACTION_RE.search(raw_text)

    #reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    reasoning = ""
    
    bracket_match = BRACKET_ACTION_RE.search(raw_text)
    action = bracket_match.group(0) if bracket_match else raw_text.strip().splitlines()[0]

    return ParsedResponse(raw_text=raw_text, reasoning=reasoning, action=action)
