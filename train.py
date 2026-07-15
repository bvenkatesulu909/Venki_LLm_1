"""Venki_llm_1 — pre-training script.

Pipeline: load corpus -> train (or reload) byte-level BPE tokenizer ->
encode corpus -> vectorized random-window dataset -> AdamW with linear
warmup + cosine decay -> save best.pt incrementally on loss improvement.

Usage:
  python train.py --vocab_size 4000 --n_layer 4 --n_head 4 --n_embd 192 \
      --block_size 128 --batch_size 64 --steps 3000
"""
import argparse
import math
import os
import time

import torch
from torch.nn import functional as F

from bpe import Tokenizer
from model import GPT, Config


class TokenDataset(torch.utils.data.Dataset):
    """Vectorized random-window sampling (pre-drawn contiguous spans)."""
    def __init__(self, ids, block_size, num_windows):
        self.block_size = block_size
        ids_t = torch.tensor(ids, dtype=torch.long)
        n, span, max_start = len(ids), block_size + 1, len(ids) - block_size - 1
        assert max_start > 0, "corpus too small for block_size"
        starts = torch.randint(0, max_start, (num_windows,))
        idx = starts.unsqueeze(1) + torch.arange(span)
        seq = ids_t[idx]
        self.x = seq[:, :block_size].clone()
        self.y = seq[:, 1:].clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--num_windows", type=int, default=120000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    # ---- corpus ----
    corpus_file = os.path.join(args.data_dir, "corpus.txt")
    assert os.path.exists(corpus_file), f"No corpus.txt found in {args.data_dir}"
    with open(corpus_file, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    nchars = len(text)
    print(f"[corpus] {nchars:,} chars (~{nchars/4/1e6:.2f}M tokens) from corpus.txt", flush=True)

    # ---- tokenizer ----
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.load(tok_path)
        print(f"[tok] loaded cached vocab={len(tok.encoder)}", flush=True)
    else:
        tok = Tokenizer(vocab_size=args.vocab_size)
        t0 = time.time()
        v = tok.train(text)
        print(f"[tok] trained vocab={v} in {time.time()-t0:.1f}s", flush=True)
        tok.save(tok_path)

    # ---- ids (cached) ----
    ids_path = os.path.join(args.out_dir, "ids.pt")
    if os.path.exists(ids_path):
        ids = torch.load(ids_path, weights_only=True)
        print(f"[ids] loaded {len(ids):,} tokens", flush=True)
    else:
        ids = tok.encode(text)
        ids = torch.tensor(ids, dtype=torch.long)
        torch.save(ids, ids_path)
        print(f"[ids] encoded {len(ids):,} tokens (~{len(ids)/1e6:.2f}M)", flush=True)

    assert len(tok.encoder) == args.vocab_size, \
        f"tokenizer vocab {len(tok.encoder)} != requested --vocab_size {args.vocab_size}"

    # ---- model ----
    cfg = Config(vocab_size=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                 n_embd=args.n_embd, block_size=args.block_size)
    model = GPT(cfg).to(args.device)
    nparams = model.num_params()
    print(f"[model] params={nparams:,} ({nparams/1e6:.2f}M)", flush=True)

    # ---- optimizer ----
    decay = set()
    no_decay = set()
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            decay.add(name)
        else:
            no_decay.add(name)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if n in decay], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    ds = TokenDataset(ids.tolist(), args.block_size, args.num_windows)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    best_loss = float("inf")
    t0 = time.time()
    step = 0
    loader = iter(dl)
    for step in range(args.steps):
        if step % len(dl) == 0:
            loader = iter(dl)
        x, y = next(loader)
        x, y = x.to(args.device), y.to(args.device)
        lr = get_lr(step, args.warmup_steps, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "step": step, "loss": best_loss},
                       os.path.join(args.out_dir, "best.pt"))
        if step % 20 == 0:
            dt = time.time() - t0
            print(f"step {step:>5d}/{args.steps} loss {loss.item():.4f} "
                  f"lr {lr:.2e} best {best_loss:.4f} [{dt:.0f}s]", flush=True)
            t0 = time.time()

    # final checkpoint
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "step": step, "loss": loss.item()},
               os.path.join(args.out_dir, "last.pt"))
    print(f"[done] final loss {loss.item():.4f} best {best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
