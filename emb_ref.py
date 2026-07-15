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
x = m.tok_emb(idx) + m.pos_emb[:, :len(ids)]
emb = x[0, 0].detach().float().tolist()
json.dump({"ids": ids, "emb0": emb}, open(os.path.join(ROOT, "deploy", "_emb_ref.json"), "w"))
print("ids", ids)
print("emb0[:6]", [round(v, 5) for v in emb[:6]])
