from pathlib import Path
from vllm import LLM
from transformers import AutoTokenizer
from completion_generator import generate_completions
from assign_vote_rewards import assign_vote_rewards

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OBS_DIR = Path("scenarios/observations_vote")          # Ordner mit den .txt-Dateien

# Modell EINMAL laden und an jeden Aufruf durchreichen
llm = LLM(model=MODEL, gpu_memory_utilization=0.6)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

for game_id, scenario in enumerate(sorted(OBS_DIR.glob("*.txt"))):
    output = f"scenarios/completions_vote/{scenario.stem}.jsonl"
    generate_completions(
        scenario=str(scenario),
        model=MODEL,
        player_id=7,
        num_completions=8,
        game_id=game_id,          # jede Datei eine eigene GRPO-Gruppe
        output=output,
        llm=llm,                  # wiederverwenden statt neu laden
        tokenizer=tokenizer,
    )
    assign_vote_rewards(output)
    print(f"[{scenario.name}] fertig -> {output}")