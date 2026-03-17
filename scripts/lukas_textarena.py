import os
from scripts.openrouterTextAgent import OpenRouterTextArenaAgent
import textarena as ta
import json

#os.environ['OPENROUTER_API_KEY'] = [API key hier einsetzen]

logs = []

# Initialize agents
agents = {
    i: OpenRouterTextArenaAgent(
        model_name="openrouter/free",
        temperature=0.8
    )
    for i in range(6)
}

# Initialize the environment
env = ta.make(env_id="SecretMafia-v0")

# wrap it for additional visualizations
env = ta.wrappers.SimpleRenderWrapper(env=env)

env.reset(num_players=len(agents))

step=0
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)["action"]
    raw_response = agents[player_id](observation)["raw_response"]
    reasoning_trace = agents[player_id](observation)["reasoning_trace"]
    done, step_info = env.step(action=action)
    logs.append({
        "step": step,
        "player_id": player_id,
        "observation": observation,
        "action": action,
        "step_info": step_info,
        "raw_response": raw_response,
        "reasoning_trace": reasoning_trace
    })

    step += 1

rewards, game_info = env.close()
game_log = {
    "num_players": len(agents),
    "steps": logs,
    "rewards": rewards,
    "game_info": game_info
}

with open("mafia_game_log.json", "w") as f:
    json.dump(game_log, f, indent=2)

print("Game finished!")
print("Rewards:", rewards)
print("Log saved to mafia_game_log.json")

with open("mafia_game_log.json") as f:
    data = json.load(f)

print(data.keys())
for step in data["steps"]:
    print(f"Player {step['player_id']}: {step['action']}")

for step in data["steps"]:
    print(f"Player {step['player_id']}: {step['raw_response']}")