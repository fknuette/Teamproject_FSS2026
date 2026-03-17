from teamproject_fss2026.textarena_utils import build_agent_prompt, parse_model_response
import textarena as ta
from textarena.core import Agent

class OpenRouterTextArenaAgent(Agent):
    def __init__(self, model_name: str, temperature: float):
        super().__init__()
        self.model_name = model_name
        
        # Use existing OpenRouter agent internally
        self.client = ta.agents.OpenRouterAgent(
            model_name=model_name,
            temperature=temperature
        )

    def __call__(self, observation: str) -> dict[str, str]:
        # 1️⃣ Build custom prompt
        prompt = observation#build_agent_prompt(observation)

        # 2️⃣ Call OpenRouter API
        raw_response = self.client(prompt)

        # 3️⃣ Parse response
        parsed = parse_model_response(raw_response)

        return {
            "prompt": prompt,
            "raw_response": parsed.raw_text,
            "reasoning_trace": parsed.reasoning,
            "action": parsed.action,
        }