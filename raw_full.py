import torch, json, os, math
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
b = m.blocks[0]
h1 = b.ln1(x)
qkv = b.attn.c_attn(h1)
B, T, C = 1, len(ids), cfg.n_embd
q, k, v = qkv.split(C, dim=2)
nh = cfg.n_head; hd = C // nh
q = q.view(B, T, nh, hd).transpose(1, 2)
k = k.view(B, T, nh, hd).transpose(1, 2)
v = v.view(B, T, nh, hd).transpose(1, 2)
scale = 1.0 / math.sqrt(hd)
att = (q @ k.transpose(-2, -1)) * scale
causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
att = att.masked_fill(causal == 0, float('-inf'))
att = torch.softmax(att, dim=-1)
y = att @ v           # [B,nh,T,hd]
y = y.transpose(1, 2).contiguous().view(B, T, C)
raw = y[0, 0].detach().float().tolist()  # full 192 vector, position 0
json.dump(raw, open(os.path.join(ROOT, "deploy", "_raw_full.json"), "w"))
print("PY raw[0,0:6]:", [round(v, 5) for v in raw[:6]])
print("PY raw[0,48:54] (head1):", [round(v, 5) for v in raw[48:54]])
