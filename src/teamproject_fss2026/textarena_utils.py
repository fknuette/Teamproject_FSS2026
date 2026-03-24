from __future__ import annotations

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
def build_agent_prompt(observation: str, voting : boolean) -> str:
    
    final_prompt = ""
    
    # Instruktion
    final_prompt += "### INSTRUCTION:\n"
    final_prompt += "You are a player in a text-based social deduction game. Your goal is to win by outsmarting other players through strategic thinking and deception. Carefully analyze the information provided and make your move accordingly.\n\n"
    
    # Context
    final_prompt += "### CONTEXT:\n"
    final_prompt += observation + "\n\n"
    
    # Task
    instruct_prompt = ""
    
    if voting:
        instruct_prompt += "Choose exactly one valid target for your action. This means after ANSWER you put just ONE Number corresponding to the player you want to target and not your own Number!\n\n"
    else:
        instruct_prompt += "You can talk and share your thoughts with other players. After ANSWER you can write freely what you want to say. But remember, the other players will see it and might use it against you, so choose your words wisely!\n\n"
    # Rules
    # Answer
    final_prompt += "### ANSWER:\n"
    '''
    if voting:
        final_prompt += "I would vote for player ["
    '''
    return final_prompt
    


def parse_model_response(raw_text: str) -> ParsedResponse:
    # reasoning_match = THINK_RE.search(raw_text)
    # action_match = ACTION_RE.search(raw_text)

    #reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    reasoning = ""
    
    bracket_match = BRACKET_ACTION_RE.search(raw_text)
    action = bracket_match.group(0) if bracket_match else raw_text.strip().splitlines()[0]

    return ParsedResponse(raw_text=raw_text, reasoning=reasoning, action=action)
