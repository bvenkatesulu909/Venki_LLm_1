"""Venki_llm_1 — Supervised Fine-Tuning (SFT) on instruction data.

Formats each example with an instruction template, masks the prompt tokens
(labels=-100) so loss is computed only on the response. Produces sft.pt.
"""
import argparse
import json
import os
import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bpe import Tokenizer
from model import GPT, Config


TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}\n"
)


def load_sft(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instr = obj.get("instruction") or obj.get("prompt") or ""
            resp = obj.get("response") or obj.get("completion") or ""
            rows.append((instr, resp))
    return rows


def build_sequences(rows, tok, block_size):
    xs, ys = [], []
    for instr, resp in rows:
        text = TEMPLATE.format(instruction=instr, response=resp)
        instr_text = TEMPLATE.format(instruction=instr, response="")
        full = tok.encode(text)
        prompt_only = tok.encode(instr_text)
        if len(full) > block_size:
            full = full[:block_size]
            prompt_only = prompt_only[:block_size]
        labels = [-100] * len(prompt_only) + full[len(prompt_only):]
        # pad
        pad = block_size - len(full)
        if pad > 0:
            full = full + [0] * pad
            labels = labels + [-100] * pad
        xs.append(full)
        ys.append(labels)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--base", default="out/best.pt")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sd = torch.load(args.base, map_location=args.device, weights_only=False)
    cfg = Config(**sd["config"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(sd["model"])
    tok = Tokenizer.load(os.path.join(os.path.dirname(args.base), "tokenizer.json"))

    rows = load_sft(args.data)
    random.shuffle(rows)
    print(f"[sft] {len(rows)} examples", flush=True)
    xs, ys = build_sequences(rows, tok, cfg.block_size)
    ds = torch.utils.data.TensorDataset(xs, ys)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        n = 0
        for x, y in dl:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        print(f"[sft] epoch {ep+1}/{args.epochs} loss {tot/max(n,1):.4f}", flush=True)

    out = os.path.join(args.out_dir, "sft.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out)
    print(f"[sft] saved {out}", flush=True)


if __name__ == "__main__":
    main()
