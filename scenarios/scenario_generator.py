"""
Interaktives Aufzeichnungs-Skript fuer Single-Step-Testszenarien.

Du spielst als Mensch alle Sitze eines SecretMafia-Spiels und konstruierst so
gezielt eine Ausgangssituation. Am Ende wird die letzte Beobachtung als .txt
gespeichert -- exakt so, wie TextArena sie ausgibt und wie der Loop sie als
Prompt bekommen wuerde.

Aufruf:
    python record_scenario.py
    python record_scenario.py --env-id SecretMafia-v0 --num-players 8 --output my_prompt.txt

Beenden mitten im Spiel: Ctrl-C -- die bis dahin letzte Beobachtung wird
trotzdem gespeichert.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import textarena as ta

SEED = 45

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mafia-Szenario interaktiv aufzeichnen")
    p.add_argument("--env-id", type=str, default="SecretMafia-v0",
                   help="TextArena Environment (Default: SecretMafia-v0)")
    p.add_argument("--num-players", type=int, default=8,
                   help="Anzahl Spieler (Default: 8)")
    p.add_argument("--output", type=str, default="",
                   help="Zielpfad fuer die .txt-Aufzeichnung. Leer -> automatischer Zeitstempel-Name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Zielordner: Unterordner "prompts" NEBEN diesem Skript (unabhaengig davon,
    # aus welchem Arbeitsverzeichnis aufgerufen wird).
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "observations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zielpfad festlegen (mit Zeitstempel, damit nichts ueberschrieben wird)
    if args.output:
        # relativer --output-Name landet im observations-Ordner; absoluter Pfad wird respektiert
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
    else:
        out_path = out_dir / f"scenario_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Alle Sitze bekommen einen menschlichen Agenten -> du steuerst alle Spieler
    agents = {pid: ta.agents.HumanAgent() for pid in range(args.num_players)}

    print("=" * 70)
    print(f"Environment : {args.env_id}")
    print(f"Spieler     : {args.num_players} (alle von dir gesteuert)")
    print(f"Ausgabe     : {out_path}")
    print("=" * 70)
    print("Tipp: Spiele bis zu dem Moment, den du einfrieren willst (z. B. bis")
    print("      direkt vor dem Zug des zu testenden Spielers), dann Ctrl-C.")
    print("=" * 70)

    env = ta.make(env_id=args.env_id)
    env.reset(num_players=len(agents), seed=SEED)

    # Wir merken uns die jeweils letzte Beobachtung (spaeter: dein fixer Prompt).
    last_observation: str | None = None

    done = False
    turn_id = 0
    try:
        while not done:
            player_id, observation = env.get_observation()

            # Klar sichtbar machen, wer gerade dran ist und was er sieht
            print("\n" + "-" * 70)
            print(f"[Turn {turn_id}] Spieler {player_id} ist am Zug.")
            print("-" * 70)
            print(observation)
            print("-" * 70)

            # Beobachtung merken, BEVOR der Zug passiert
            last_observation = observation

            action = agents[player_id](observation)
            done, step_info = env.step(action=action)
            turn_id += 1

    except KeyboardInterrupt:
        # Bewusstes Abbrechen ist der Normalfall: du willst ja nur BIS zu einem
        # bestimmten Punkt spielen. Die letzte Beobachtung bleibt erhalten.
        print("\n\n[Abbruch] Ctrl-C erkannt -- speichere die letzte Beobachtung.")

    else:
        # Spiel regulaer zu Ende gespielt -> Rewards nur anzeigen
        rewards, game_info = env.close()
        print("\n" + "=" * 70)
        print(f"Spiel beendet. Rewards: {rewards}")
        print(f"Game Info: {game_info}")
        print("=" * 70)

    # Letzte Beobachtung als reinen Text wegschreiben
    if last_observation is None:
        raise SystemExit("[Fehler] Keine einzige Beobachtung aufgezeichnet.")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(last_observation)

    print(f"\n[Gespeichert] Letzte Beobachtung -> {out_path}")


if __name__ == "__main__":
    main()