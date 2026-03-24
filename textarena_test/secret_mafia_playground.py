import os
import textarena as ta
from textarena.core import Agent
import re
import random

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

def clear():
    os.system("cls" if os.name == "nt" else "clear")

class RandomSecretMafiaAgent(Agent):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)  # Use an independent RNG for this agent

    def __call__(self, observation):  
        move = self._determine_action(observation)
        print("\n\n+++ +++ +++")
        print("Observation:\n", observation)
        print("Chosen action:", move)        
        return f"{move}"

    def _determine_action(self, observation):
        """
        Determines the action based on the last occurrence of specific patterns in the observation string.

        Args:
            observation (str): The input string containing the game observation.

        Returns:
            str: "hello" if the last pattern is "Day breaks. Discuss for 3 rounds, then a vote will follow."
                 One of the valid targets (e.g., "[1]") if the last pattern is "Valid targets: [2], [3], [5]".
                 One of the valid votes (e.g., "[1]") if the last pattern is "Voting phase - submit one vote in format [X]. Valid: [1], [2], [3], [4], [5]".
                 One of the valid players to protect (e.g., "[0]") if the last pattern is "choose one player to protect: [0], [1], [3], [4], [5]".
                 One of the valid players to investigate (e.g., "[1]") if the last pattern is "choose one player to investigate: [1], [2], [3], [4], [5]".
                 None if no valid action can be determined.
        """
        # Define the patterns
        pattern_discuss = r"Day breaks\. Discuss for 3 rounds, then a vote will follow\."
        pattern_targets = r"Valid targets: (\[\d+\](?:, \[\d+\])*)"
        pattern_voting = r"Voting phase - submit one vote in format \[X\]\. Valid: (\[\d+\](?:, \[\d+\])*)"
        pattern_protect = r"choose one player to protect: (\[\d+\](?:, \[\d+\])*)"
        pattern_investigate = r"choose one player to investigate: (\[\d+\](?:, \[\d+\])*)"

        # Helper function to find the last match of a pattern
        def find_last_match(pattern, observation):
            matches = list(re.finditer(pattern, observation))
            return matches[-1] if matches else None

        # Find the last occurrence of each pattern
        discuss_match = find_last_match(pattern_discuss, observation)
        targets_match = find_last_match(pattern_targets, observation)
        voting_match = find_last_match(pattern_voting, observation)
        protect_match = find_last_match(pattern_protect, observation)
        investigate_match = find_last_match(pattern_investigate, observation)

        # Collect all matches with their positions and actions
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

        # If no matches are found, return None
        if not matches:
            return None

        # Sort matches by their position in the string (end position), and take the last one
        matches.sort(key=lambda x: x[0], reverse=True)
        last_match = matches[0]

        # Determine the action based on the last match
        action = last_match[1]
        if action == "hello":
            return self.rng.choice(DUMMY_SECRET_MAFIA_VAGUE_LINES)
        elif action == "kill":
            # Extract valid targets and return one of them
            targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(targets) if targets else None
        elif action == "vote":
            # Extract valid votes and return one of them
            votes = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(votes) if votes else None
        elif action == "protect":
            # Extract valid players to protect and return one of them
            protect_targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(protect_targets) if protect_targets else None
        elif action == "investigate":
            # Extract valid players to investigate and return one of them
            investigate_targets = re.findall(r"\[\d+\]", last_match[2])
            return self.rng.choice(investigate_targets) if investigate_targets else None

        # Default case: No valid action
        return None

def main():
    num_players = 6  # SecretMafia-v0 erwartet 6-15
    env = ta.make("SecretMafia-v0")

    agents = {pid: RandomSecretMafiaAgent() for pid in range(num_players-1)}
    agents[num_players-1] = ta.agents.HumanAgent()
    env.reset(num_players=num_players)

    done = False
    while not done:
        player_id, obs = env.get_observation()
        clear()
        print("=" * 60)
        print(f"You are Player {player_id}. (Please only you look at the screen!)")
        print("=" * 60)
        action = agents[player_id](obs)   
        done, info = env.step(action=action)

    rewards, game_info = env.close()
    print("\nGame Over")
    print("Rewards:", rewards)
    print("Game info:", game_info)

if __name__ == "__main__":
    main()
    