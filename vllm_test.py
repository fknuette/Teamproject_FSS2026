from vllm import LLM, SamplingParams
import time
import gc

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=False,   # erstmal stabiler zum Testen
)

prompt = "Explain briefly what an LLM is."
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
)

outputs = llm.generate([prompt], sampling_params)

for out in outputs:
    print(out.outputs[0].text)

time.sleep(2)   # kurzer Puffer vor dem Beenden
del llm
gc.collect()
time.sleep(2)
