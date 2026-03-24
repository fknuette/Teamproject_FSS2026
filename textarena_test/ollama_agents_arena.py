import textarena as ta
from textarena.core import Agent
#from secret_mafia_playground import RandomSecretMafiaAgent

def main():
    num_players = 6
    env = ta.make("SecretMafia-v0")

    agents = {pid: ta.agents.OllamaAgent(model_name="llama3") for pid in range(num_players)}
    env.reset(num_players=num_players)

    log = []  # Spielverlauf sammeln
    
    done = False
    while not done:
        player_id, obs = env.get_observation()
        action = agents[player_id](obs)
        
        # Jeden Zug loggen
        log.append({
            "player_id": player_id,
            "observation": obs,
            "action": action
        })
        
        done, info = env.step(action=action)

    rewards, game_info = env.close()
    
    # Log speichern
    import json
    from datetime import datetime
    
    filename = f"mafia_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({
            "log": log,
            "rewards": rewards,
            "game_info": game_info
        }, f, indent=2, default=str)
    
    print(f"Spielverlauf gespeichert in: {filename}")

if __name__ == "__main__":
    main()
