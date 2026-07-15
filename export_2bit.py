"""Export Venki_llm_1 (2-bit quantized) weights to browser JSON.

The big 2-D matrices are stored as packed 2-bit codes + per-group scale/min
(float16). The JS port dequantizes them per group at load time. LayerNorm /
1-D params are stored verbatim as float lists. Resulting model.json is ~1.5MB.
"""
import json
import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DEPLOY = os.path.join(ROOT, "deploy")


def main():
    ck = torch.load(os.path.join(OUT, "venki_llm_2bit.pt"), map_location="cpu", weights_only=False)
    assert ck.get("quantized"), "expected quantized checkpoint"
    cfg = ck["config"]
    state = ck["model"]

    weights = {}
    for k, v in state.items():
        if isinstance(v, dict) and "__q__" in v:
            d = v["__q__"]
            packed = d["packed"].tolist()           # list[int] 4 codes/byte
            scale = [float(x) for x in d["scale"].tolist()]
            mn = [float(x) for x in d["min"].tolist()]
            weights[k] = {"q": packed, "scale": scale, "min": mn,
                          "shape": d["shape"], "pad": d["pad"], "group": d["group"]}
        else:
            weights[k] = {"f": torch.tensor(v).detach().cpu().float().reshape(-1).tolist()}
    blob = {"config": cfg, "quantized": True, "weights": weights}
    os.makedirs(DEPLOY, exist_ok=True)
    with open(os.path.join(DEPLOY, "model.json"), "w") as f:
        json.dump(blob, f)
    size = os.path.getsize(os.path.join(DEPLOY, "model.json")) / 1e6
    print(f"model.json (2-bit) written: {size:.2f} MB, {len(weights)} tensors")


if __name__ == "__main__":
    main()
