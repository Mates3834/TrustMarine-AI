"""Run a REAL local Ollama model; no simulated model answers."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from src.core import generate_data
from src.llm_agent import build_demo_agent, OllamaClient

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model',default='qwen2.5:7b')
 p.add_argument('--ollama-url',default='http://127.0.0.1:11434')
 p.add_argument('--fault',choices=['normal','cooling_degradation','bearing_wear','lubrication_loss','load_instability'],default='bearing_wear')
 p.add_argument('--output',default='results/llm_run.json')
 a=p.parse_args()
 client=OllamaClient(model=a.model,url=a.ollama_url)
 agent=build_demo_agent(client)
 sample=generate_data(n_per_class=100,seed=91)
 sample=sample[sample.fault_type==a.fault].iloc[0].to_dict()
 result=agent.run(sample)
 payload={'synthetic_ground_truth':a.fault,'sample':sample,'result':asdict(result)}
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
 print(result.answer)
 print('\nSaved audit trail:',output)
if __name__=='__main__': main()
