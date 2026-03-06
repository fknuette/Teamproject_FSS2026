import json
import sys

def convert_log(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    lines = []
    
    # Rollen anzeigen
    lines.append("=== ROLLEN ===")
    for pid, info in data["game_info"].items():
        lines.append(f"Spieler {pid}: {info['role']}")
    
    # Spielverlauf
    lines.append("\n=== SPIELVERLAUF ===")
    for i, entry in enumerate(data["log"]):
        pid = entry["player_id"]
        action = entry["action"]
        lines.append(f"\n--- Zug {i+1}: Spieler {pid} ---")
        lines.append(action)
    
    # Ergebnis
    lines.append("\n=== ERGEBNIS ===")
    for pid, reward in data["rewards"].items():
        result = "Gewonnen" if reward == 1 else "Verloren"
        lines.append(f"Spieler {pid}: {result}")
    
    lines.append(f"\n{data['game_info']['0']['reason']}")
    
    # In txt-Datei speichern
    output_path = filepath.replace(".json", ".txt")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Gespeichert: {output_path}")

if __name__ == "__main__":
    convert_log("mafia_game_20260306_145518.json")