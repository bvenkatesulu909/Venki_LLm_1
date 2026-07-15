import torch, json, os
from bpe import Tokenizer
from model import GPT, Config
ROOT = os.path.dirname(os.path.abspath(__file__))
ck = torch.load(os.path.join(ROOT, "out", "venki_llm_2bit.pt"), map_location="cpu", weights_only=False)
cfg = Config(**ck["config"])

def dq(sd):
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and "__q__" in v:
            d = v["__q__"]; p = d["packed"].to(torch.int64)
            q = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], 1).reshape(-1, d["group"]).float()
            w = q * d["scale"].float().unsqueeze(1) + d["min"].float().unsqueeze(1)
            w = w.reshape(-1)
            if d["pad"]: w = w[:-d["pad"]]
            out[k] = w.reshape(d["shape"])
        else:
            out[k] = v if torch.is_tensor(v) else torch.tensor(v)
    return out

m = GPT(cfg).eval(); m.load_state_dict(dq(ck["model"]))
tok = Tokenizer.load(os.path.join(ROOT, "out", "tokenizer.json"))
ids = tok.encode("Venki_llm_1 is a small")
idx = torch.tensor(ids).unsqueeze(0)
# manual block-0 forward to capture intermediate
emb = m.tok_emb(idx) + m.pos_emb[:, :len(ids)]
x = emb
b = m.blocks[0]
h1 = b.ln1(x)
a = b.attn(h1)
x2 = x + a
h2 = b.ln2(x2)
mlp_out = b.mlp(h2)
x3 = x2 + mlp_out
out = {"ln1_0": h1[0, 0].detach().float().tolist()[:6],
       "attn_0": a[0, 0].detach().float().tolist()[:6],
       "after_attn_0": x2[0, 0].detach().float().tolist()[:6],
       "ln2_0": h2[0, 0].detach().float().tolist()[:6],
       "mlp_0": mlp_out[0, 0].detach().float().tolist()[:6],
       "block0_out_0": x3[0, 0].detach().float().tolist()[:6]}
json.dump(out, open(os.path.join(ROOT, "deploy", "_blk_ref.json"), "w"))
print(json.dumps({k: [round(v, 5) for v in vv] for k, vv in out.items()}, indent=0))
