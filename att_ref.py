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
emb = m.tok_emb(idx) + m.pos_emb[:, :len(ids)]
b = m.blocks[0]
qkv = b.attn.c_attn(emb)  # [B,T,3C]
q = qkv[0, :, :cfg.n_embd]
k = qkv[0, :, cfg.n_embd:2*cfg.n_embd]
v = qkv[0, :, 2*cfg.n_embd:]
# python attention
import torch.nn.functional as F
import math
B,T,C = 1, len(ids), cfg.n_embd
nh, hd = cfg.n_head, C//cfg.n_head
def reshape(t): return t.view(B,T,nh,hd).transpose(1,2)
qq, kk, vv = reshape(q), reshape(k), reshape(v)
att = (qq @ kk.transpose(-2,-1)) * (1/math.sqrt(hd))
caus = torch.tril(torch.ones(T,T)).view(1,1,T,T)
att = att.masked_fill(caus==0, float('-inf'))
att = F.softmax(att, -1)
yy = (att @ vv).transpose(1,2).contiguous().view(B,T,C)
out = {"qkv0": qkv[0,0].detach().float().tolist()[:6],
       "attn_out_0": yy[0,0].detach().float().tolist()[:6],
       "q0_head0": qq[0,0,0].detach().float().tolist()[:6],
       "k0_head0": kk[0,0,0].detach().float().tolist()[:6]}
json.dump(out, open(os.path.join(ROOT,"deploy","_att_ref.json"),"w"))
print("qkv0",[round(x,5) for x in out["qkv0"]])
print("q0h0",[round(x,5) for x in out["q0_head0"]])
print("attn_out0",[round(x,5) for x in out["attn_out_0"]])
