"""Venki_llm_1 — inference / text generation."""
import argparse
import os
import torch
from bpe import Tokenizer
from model import GPT, Config


def load_model(out_dir, ckpt="best.pt", device="cpu"):
    sd = torch.load(os.path.join(out_dir, ckpt), map_location=device, weights_only=False)
    cfg = Config(**sd["config"])
    model = GPT(cfg).to(device)
    state = sd["model"]
    if sd.get("quantized"):
        # dequantize 2-bit group-quantized tensors before loading
        deq = {}
        for k, v in state.items():
            if isinstance(v, dict) and "__q__" in v:
                # inline dequant (mirror quantize.py)
                packed = v["__q__"]["packed"].to(torch.int64)
                q = torch.stack([packed & 3, (packed >> 2) & 3,
                                 (packed >> 4) & 3, (packed >> 6) & 3], dim=1)
                q = q.reshape(-1, v["__q__"]["group"]).float()
                w = q * v["__q__"]["scale"].float().unsqueeze(1) + v["__q__"]["min"].float().unsqueeze(1)
                w = w.reshape(-1)
                if v["__q__"]["pad"]:
                    w = w[:-v["__q__"]["pad"]]
                deq[k] = w.reshape(v["__q__"]["shape"])
            else:
                deq[k] = v
        state = deq
    model.load_state_dict(state)
    model.eval()
    tok = Tokenizer.load(os.path.join(out_dir, "tokenizer.json"))
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--prompt", default="Venki_llm_1 is")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, tok = load_model(args.out_dir, args.ckpt, args.device)
    ids = tok.encode(args.prompt)
    if not ids:
        ids = [0]
    ctx = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
    with torch.no_grad():
        out_ids = model.generate(ctx, args.max_new_tokens,
                                 temperature=args.temperature,
                                 top_k=args.top_k, top_p=args.top_p)
    text = tok.decode(out_ids[0].tolist())
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
