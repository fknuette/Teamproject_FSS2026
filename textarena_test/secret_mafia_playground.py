from __future__ import annotations

import os
import re
import json
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import textarena as ta
from textarena.core import Agent


DUMMY_SECRET_MAFIA_VAGUE_LINES = [
    "I’m not sure yet—let’s gather more info first.",
    "I don’t have a strong read right now.",
    "Can we recap what happened last round?",
    "Let’s slow down and compare everyone’s claims.",
    "I’d like to hear more reasoning before we vote.",
    "I’m keeping my options open for now.",
    "Nothing conclusive from me at the moment.",
    "We should focus on contradictions, not vibes.",
    "Can everyone explain their vote from last round?",
    "Let’s avoid rushing into a random vote.",
    "I’m more interested in who is pushing hard and why.",
    "We should check for inconsistencies in stories.",
    "I’d prefer to hear from quieter voices too.",
    "Let’s do a quick summary of current suspicions.",
    "I don’t want to reveal anything prematurely.",
    "If someone has strong info, they should share carefully.",
    "I’m open to changing my mind with better evidence.",
    "I’m not convinced by the arguments so far.",
    "Let’s keep role claims minimal unless necessary.",
    "I’m watching for overconfident accusations.",
    "We should be careful about bandwagon voting.",
    "I think we need a clearer plan for voting.",
    "What’s the most suspicious pattern so far?",
    "I want to understand the logic behind the suspicions.",
    "Let’s align on what evidence we actually have.",
    "I’m okay waiting one more round to confirm patterns.",
    "Let’s talk through the timeline step by step.",
    "I’m skeptical of confident statements without support.",
    "Let’s keep discussion structured and objective.",
    "I don’t have enough to confidently accuse anyone yet.",
]


@dataclass
class TurnRecord:
    game_id: int
    observation: Any
    response: str
    reward: float = 0.0
    player_id: int = 0
    turn_id: int = 0


def _normalize_rewards(rewards: Any) -> dict[int, float]:
    if isinstance(rewards, dict):
        return {int(k): float(v) for k, v in rewards.items()}
    if isinstance(rewards, (list, tuple)):
        return {i: float(v) for i, v in enumerate(rewards)}
    raise ValueError(f"Unsupported rewards format: {type(rewards)}")


def serialize_observation(observation: Any) -> Any:
    if isinstance(observation, list):
        out = []
        for item in observation:
            if isinstance(item, tuple) and len(item) == 3:
                from_id, message, obs_type = item
                out.append(
                    {
                        "from_id": from_id,
                        "message": message,
                        "observation_type": getattr(obs_type, "name", str(obs_type)),
                    }
                )
            else:
                out.append(item)
        return out
    return observation


def clear():
    os.system("cls" if os.name == "nt" else "clear")


class RandomSecretMafiaAgent(Agent):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def __call__(self, observation):
        move = self._determine_action(observation)
        print("\n\n+++ +++ +++")
        print("Observation:\n", observation)
        print("Chosen action:", move)
        return f"{move}"

    def _determine_action(self, observation):
        pattern_discuss = r"Day breaks\. Discuss for 3 rounds, then a vote will follow\."
        pattern_targets = r"Valid targets: (\[\d+\](?:, \[\d+\])*)"
        pattern_voting = r"Voting phase - submit one vote in format \[X\]\. Valid: (\[\d+\](?:, \[\d+\])*)"
        pattern_protect = r"choose one player to protect: (\[\d+\](?:, \[\d+\])*)"
        pattern_investigate = r"choose one player to investigate: (\[\d+\](?:, \[\d+\])*)"

        def find_last_match(pattern, observation):
            text = str(observation)
            matches = list(re.finditer(pattern, text))
            return matches[-1] if matches else None

        discuss_match = find_last_match(pattern_discuss, observation)
        targets_match = find_last_match(pattern_targets, observation)
        voting_match = find_last_match(pattern_voting, observation)
        protect_match = find_last_match(pattern_protect, observation)
        investigate_match = find_last_match(pattern_investigate, observation)

        matches = []
        if discuss_match:
            matches.append((discuss_match.end(), "hello"))
        if targets_match:
            matches.append((targets_match.end(), "kill", targets_match.group(1)))
        if voting_match:
            matches.append((voting_match.end(), "vote", voting_match.group(1)))
        if protect_match:
            matches.append((protect_match.end(), "protect", protect_match.group(1)))
        if investigate_match:
            matches.append((investigate_match.end(), "investigate", investigate_match.group(1)))

        if not matches:
            return None

        matches.sort(key=lambda x: x[0], reverse=True)
        last_match = matches[0]

        action = last_match[1]
        if action == "hello":
            return self.rng.choice(DUMMY_SECRET_MAFIA_VAGUE_LINES)
        elif action == "kill":
            targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(targets) if targets else None
        elif action == "vote":
            votes = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(votes) if votes else None
        elif action == "protect":
            protect_targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(protect_targets) if protect_targets else None
        elif action == "investigate":
            investigate_targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(investigate_targets) if investigate_targets else None

        return None


# ===== NEW CODE START: invalid-move helpers =====

ATTEMPTED_INVALID_AT_END_RE = re.compile(
    r"\[GAME\] Player (?P<pid>\d+) attempted an invalid move\. "
    r"Reason: .*?Please resubmit a valid move and remember to follow the game rules to avoid penalties\.\s*$",
    re.DOTALL,
)

ELIMINATED_INVALID_RE = re.compile(
    r"^\[GAME\] Player (?P<pid>\d+) has been eliminated by making an invalid move\.\.?$"
)


def _extract_invalid_pid_from_attempted_at_end(observation: str) -> int | None:
    match = ATTEMPTED_INVALID_AT_END_RE.search(observation)
    return int(match.group("pid")) if match else None


def _extract_eliminated_invalid_lines(observation: str) -> set[str]:
    text = str(observation)
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and ELIMINATED_INVALID_RE.match(line.strip())
    }


def _extract_invalid_pid_from_eliminated_line(line: str) -> int | None:
    match = ELIMINATED_INVALID_RE.match(line)
    return int(match.group("pid")) if match else None


def _mark_last_turn_invalid(
    game_records: list[TurnRecord],
    invalid_turn_ids: set[int],
    player_id: int,
) -> None:
    for rec in reversed(game_records):
        if rec.player_id == player_id:
            invalid_turn_ids.add(rec.turn_id)
            return

# ===== NEW CODE END: invalid-move helpers =====


def main():
    game_id = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"data/playground_secret_mafia_{timestamp}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = ta.make("SecretMafia-v0")

    num_players = 7

    agents = {pid: RandomSecretMafiaAgent() for pid in range(num_players)}
    agents[2] = ta.agents.HumanAgent()

    env.reset(num_players=num_players)

    done = False
    turn_id = 0
    game_records: list[TurnRecord] = []

    # ===== NEW CODE START: invalid-turn tracking =====
    invalid_turn_ids: set[int] = set()
    seen_eliminated_lines_by_player: dict[int, set[str]] = {pid: set() for pid in agents}
    # ===== NEW CODE END: invalid-turn tracking =====

    while not done:
        player_id, obs = env.get_observation()

        # ===== NEW CODE START: attempted invalid only if at end of full observation =====
        obs_text = str(obs)
        attempted_invalid_pid = _extract_invalid_pid_from_attempted_at_end(obs_text)
        if attempted_invalid_pid is not None:
            _mark_last_turn_invalid(game_records, invalid_turn_ids, attempted_invalid_pid)
        # ===== NEW CODE END: attempted invalid only if at end of full observation =====

        # ===== NEW CODE START: eliminated invalid only if newly seen for this player =====
        current_eliminated_lines = _extract_eliminated_invalid_lines(obs_text)
        new_eliminated_lines = current_eliminated_lines - seen_eliminated_lines_by_player[player_id]

        for line in new_eliminated_lines:
            eliminated_pid = _extract_invalid_pid_from_eliminated_line(line)
            if eliminated_pid is not None:
                _mark_last_turn_invalid(game_records, invalid_turn_ids, eliminated_pid)

        seen_eliminated_lines_by_player[player_id] = current_eliminated_lines
        # ===== NEW CODE END: eliminated invalid only if newly seen for this player =====

        clear()
        print("=" * 60)
        print(f"You are Player {player_id}. (Please only you look at the screen!)")
        print("=" * 60)

        action = agents[player_id](obs)
        done, info = env.step(action=action)

        game_records.append(
            TurnRecord(
                game_id=game_id,
                turn_id=turn_id,
                player_id=player_id,
                observation=serialize_observation(obs),
                response="" if action is None else str(action),
            )
        )
        turn_id += 1

    rewards, game_info = env.close()
    reward_map = _normalize_rewards(rewards)

    for record in game_records:
        record.reward = reward_map.get(record.player_id, 0.0)

        # ===== NEW CODE START: invalid turns always get reward -1 =====
        if record.turn_id in invalid_turn_ids:
            record.reward = -1.0
        # ===== NEW CODE END: invalid turns always get reward -1 =====

    with output_path.open("w", encoding="utf-8") as f:
        for rec in game_records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print("\nGame Over")
    print("Rewards:", rewards)
    print("Game info:", game_info)
    print(f"Log written to: {output_path}")


if __name__ == "__main__":
    main()