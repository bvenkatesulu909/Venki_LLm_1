"""Dump PyTorch reference for JS parity check.

Writes _ref.json containing:
  - the probe prompt and its token ids (Python tokenizer)
  - last-position logits from the 2-bit-dequantized model
  - a Python decode of generated ids for cross-check
"""
import json
import os
import torch

from bpe import Tokenizer
from model import GPT, Config

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DEPLOY = os.path.join(ROOT, "deploy")

PROMPT = "Venki_llm_1 is a small"


def dequant_state(sd):
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and "__q__" in v:
            d = v["__q__"]
            packed = d["packed"].to(torch.int64)
            q = torch.stack([packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3], dim=1)
            q = q.reshape(-1, d["group"]).float()
            w = q * d["scale"].float().unsqueeze(1) + d["min"].float().unsqueeze(1)
            w = w.reshape(-1)
            if d["pad"]:
                w = w[:-d["pad"]]
            out[k] = w.reshape(d["shape"])
        else:
            out[k] = v if torch.is_tensor(v) else torch.tensor(v)
    return out


def main():
    ck = torch.load(os.path.join(OUT, "venki_llm_2bit.pt"), map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = GPT(cfg).eval()
    model.load_state_dict(dequant_state(ck["model"]))
    tok = Tokenizer.load(os.path.join(OUT, "tokenizer.json"))

    ids = tok.encode(PROMPT)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(idx)
    last = logits[0, -1].float()
    ref = {
        "prompt": PROMPT,
        "ids": ids,
        "last_logits": last.tolist(),
        "argmax": int(last.argmax()),
    }
    with open(os.path.join(DEPLOY, "_ref.json"), "w") as f:
        json.dump(ref, f)
    print("ref ids:", ids)
    print("argmax token:", ref["argmax"], "->", repr(tok.decode([ref["argmax"]])))
    print("wrote _ref.json")


if __name__ == "__main__":
    main()
