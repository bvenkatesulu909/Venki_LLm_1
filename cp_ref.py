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
b = m.blocks[0]
h1 = b.ln1(x)
qkv = b.attn.c_attn(h1)
C = cfg.n_embd
# raw attention
import torch.nn.functional as F, math
nh, hd = cfg.n_head, C//cfg.n_head
def rs(t): return t.view(1, len(ids), nh, hd).transpose(1, 2)
qq, kk, vv = rs(qkv[0,:, :C]), rs(qkv[0,:, C:2*C]), rs(qkv[0,:, 2*C:])
att = (qq @ kk.transpose(-2,-1)) * (1/math.sqrt(hd))
caus = torch.tril(torch.ones(len(ids), len(ids))).view(1,1,len(ids),len(ids))
att = att.masked_fill(caus==0, float('-inf'))
att = F.softmax(att, -1)
yy = (att @ vv).transpose(1,2).contiguous().view(1, len(ids), C)
# projected
proj = b.attn.c_proj(yy)
raw = yy[0,0].detach().float().tolist()[:6]
prj = proj[0,0].detach().float().tolist()[:6]
# dequantized c_proj weight row 0
cp = dq({k:v for k,v in ck['model'].items() if k=='blocks.0.attn.c_proj.weight'})['blocks.0.attn.c_proj.weight']
cp_row0 = cp[0].tolist()[:6]
json.dump({"raw0": raw, "prj0": prj, "cp_row0": cp_row0}, open(os.path.join(ROOT,"deploy","_cp_ref.json"),"w"))
print("raw attn[0,0:6]:", [round(v,5) for v in raw])
print("proj[0,0:6]   :", [round(v,5) for v in prj])
print("c_proj_w row0 :", [round(v,5) for v in cp_row0])
