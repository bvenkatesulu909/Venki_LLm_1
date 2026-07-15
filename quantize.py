"""Venki_llm_1 — Stage 5: from-scratch 2-bit group quantization.

Quantizes the big 2-D weight matrices of a trained GPT checkpoint from
float32 (32 bits) to 2 bits (4 levels {0,1,2,3}) with a per-group
scale+min. Tiny 1-D params (LayerNorm, bias, positional emb) stay float.
"""
import argparse
import os

import torch

BITS = 2
LEVELS = (1 << BITS) - 1   # 3  -> codes {0,1,2,3}
GROUP = 64                 # weights per quantization group


def quantize_tensor(w, group_size=GROUP):
    flat = w.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    g = flat.view(-1, group_size)
    mn = g.min(dim=1, keepdim=True).values
    mx = g.max(dim=1, keepdim=True).values
    scale = (mx - mn).clamp(min=1e-8) / LEVELS
    q = torch.round((g - mn) / scale).clamp(0, LEVELS).to(torch.uint8)
    q = q.view(-1, 4)
    packed = (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).to(torch.uint8)
    return {"packed": packed, "scale": scale.squeeze(1).half(),
            "min": mn.squeeze(1).half(), "shape": list(w.shape),
            "pad": pad, "group": group_size}


def dequantize_tensor(d):
    packed = d["packed"].to(torch.int64)
    q = torch.stack([packed & 3, (packed >> 2) & 3,
                     (packed >> 4) & 3, (packed >> 6) & 3], dim=1)
    q = q.reshape(-1, d["group"]).float()
    w = q * d["scale"].float().unsqueeze(1) + d["min"].float().unsqueeze(1)
    w = w.reshape(-1)
    if d["pad"]:
        w = w[:-d["pad"]]
    return w.reshape(d["shape"])


def quantize_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if v.dim() == 2 and min(v.shape) >= 16:    # PITFALL: else under `for`, not `if`
            out[k] = {"__q__": quantize_tensor(v)}
        else:
            out[k] = v
    return out


def dequantize_state_dict(sd):
    out = {}
    for k, v in sd.items():
        out[k] = dequantize_tensor(v["__q__"]) if isinstance(v, dict) and "__q__" in v else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--save", default="venki_llm_2bit.pt")
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.out_dir, args.ckpt), map_location="cpu", weights_only=False)
    fp32 = ck["model"]
    qsd = quantize_state_dict(fp32)
    torch.save({"model": qsd, "config": ck["config"], "quantized": True},
               os.path.join(args.out_dir, args.save))
    orig_bytes = sum(v.numel() * 4 for v in fp32.values())
    q_bytes = 0
    for v in qsd.values():
        if isinstance(v, dict) and "__q__" in v:
            d = v["__q__"]
            q_bytes += d["packed"].numel() + d["scale"].numel() * 2 + d["min"].numel() * 2
        else:
            q_bytes += v.numel() * 4
    deq = dequantize_state_dict(qsd)
    errs = [(fp32[k] - deq[k]).abs().mean().item()
            for k in fp32 if fp32[k].dim() == 2 and min(fp32[k].shape) >= 16]
    print(f"original: {orig_bytes/1e6:.3f} MB (float32)")
    print(f"2-bit   : {q_bytes/1e6:.3f} MB")
    print(f"compression: {orig_bytes/q_bytes:.2f}x")
    print(f"mean abs recon error: {sum(errs)/len(errs):.5f}")
    print(f"saved {args.out_dir}/{args.save}")


if __name__ == "__main__":
    main()
