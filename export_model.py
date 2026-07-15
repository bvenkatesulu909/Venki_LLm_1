"""Export Venki_llm_1 weights to browser-friendly JSON for a pure-JS port.

Produces:
  deploy/model.json  — config + all weight matrices as nested lists (float32)
  deploy/tokenizer.json — already exists; copied
The JS forward pass (app.js) loads these and runs the GPT without any backend.
"""
import json
import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DEPLOY = os.path.join(ROOT, "deploy")


def to_list(t):
    return t.detach().cpu().float().tolist()


def main():
    sd = torch.load(os.path.join(OUT, "best.pt"), map_location="cpu", weights_only=False)
    cfg = sd["config"]
    state = sd["model"]
    weights = {k: to_list(v) for k, v in state.items()}
    blob = {"config": cfg, "weights": weights}
    os.makedirs(DEPLOY, exist_ok=True)
    with open(os.path.join(DEPLOY, "model.json"), "w") as f:
        json.dump(blob, f)
    size = os.path.getsize(os.path.join(DEPLOY, "model.json")) / 1e6
    print(f"model.json written: {size:.2f} MB, {len(weights)} tensors")
    print("config:", cfg)


if __name__ == "__main__":
    main()
