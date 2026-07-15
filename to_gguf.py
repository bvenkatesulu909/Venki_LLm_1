"""Convert Venki_llm_1 (out/best.pt) -> a GGUF file loadable by llama.cpp / Ollama.

Architecture maps onto llama.cpp's `gpt2` family. CRITICAL: llama.cpp's gpt2
expects every 2D LINEAR weight in [in_features, out_features] layout
(i.e. transposed vs PyTorch's [out, in]). We verified this against a known-good
gpt2 GGUF (reference `venki-agent-llm`): its on-disk tensors are
  attn_qkv.weight [C, 3C], ffn_up.weight [C, 4C], ffn_down.weight [4C, C],
  token_embd.weight [C, V], position_embd.weight [C, B], output.weight [C, V].
So we store PyTorch weights with .T to hit those exact on-disk shapes.
LN weights/biases and 1D bias tensors are not transposed.
"""
import os, json, argparse, math
import torch
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, GGUFEndian

def load_checkpoint(src):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd, cfg = ck["model"], ck.get("config", None)
    else:
        sd, cfg = ck, None
    if cfg is None:
        raise SystemExit("checkpoint missing config; pass --vocab etc.")
    return sd, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/best.pt")
    ap.add_argument("--tok", default="out/tokenizer.json")
    ap.add_argument("--out", default="venki_llm_1.gguf")
    ap.add_argument("--name", default="venki_llm_1")
    args = ap.parse_args()

    sd, cfg = load_checkpoint(args.src)
    V = int(cfg["vocab_size"]); L = int(cfg["n_layer"]); H = int(cfg["n_head"])
    C = int(cfg["n_embd"]); B = int(cfg["block_size"])
    print(f"model: vocab={V} layers={L} heads={H} n_embd={C} block_size={B}")

    tok = json.load(open(args.tok, "r", encoding="utf-8"))
    encoder = tok["encoder"]           # glyph -> id
    merges = tok["merges"]             # [[a, b], ...] glyph pairs
    decoder = {int(v): k for k, v in encoder.items()}

    f = open(args.out, "wb")
    gw = GGUFWriter(args.out, arch="gpt2", endianess=GGUFEndian.LITTLE)

    # ---- general metadata ----
    gw.add_name(args.name)
    gw.add_author("Venkatesh (bvenkatesulu909)")
    gw.add_description("Venki_llm_1: a 2.54M-param GPT-style LLM trained from scratch.")
    gw.add_context_length(B)
    gw.add_embedding_length(C)
    gw.add_block_count(L)
    gw.add_feed_forward_length(4 * C)
    gw.add_head_count(H)
    gw.add_head_count_kv(H)
    gw.add_layer_norm_eps(1e-5)
    gw.add_causal_attention(True)
    gw.add_file_type(0)  # 0 = all F32

    # ---- tensors ----
    # NOTE (verified): GGUFWriter TRANSPOSES 2D tensors on write.
    # A PyTorch [out,in] weight written as-is lands on-disk as [in,out],
    # which is exactly what llama.cpp's gpt2 loader expects. So we do NOT .T.
    def add(name, t, transpose=False):
        arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy())
        if transpose and arr.ndim == 2:
            arr = arr.T
        gw.add_tensor(name, arr)
        print("  +", name, tuple(arr.shape))

    # token + positional embeddings -> [C, V] / [C, B] on-disk via GGUFWriter transpose
    add("token_embd.weight", sd["tok_emb.weight"])        # [V,C]->[C,V] on-disk
    add("position_embd.weight", sd["pos_emb"].squeeze(0))  # [B,C]->[C,B] on-disk

    for i in range(L):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", sd[f"blocks.{i}.ln1.weight"])
        add(p + "attn_norm.bias",   sd[f"blocks.{i}.ln1.bias"])
        add(p + "attn_qkv.weight",  sd[f"blocks.{i}.attn.c_attn.weight"])  # [3C,C]->[C,3C] on-disk
        add(p + "attn_qkv.bias",    torch.zeros(3 * C))                     # required by loader
        add(p + "attn_output.weight", sd[f"blocks.{i}.attn.c_proj.weight"])  # [C,C]->[C,C] on-disk
        add(p + "attn_output.bias",   torch.zeros(C))
        add(p + "ffn_norm.weight",   sd[f"blocks.{i}.ln2.weight"])
        add(p + "ffn_norm.bias",     sd[f"blocks.{i}.ln2.bias"])
        add(p + "ffn_up.weight",     sd[f"blocks.{i}.mlp.c_fc.weight"])    # [4C,C]->[C,4C] on-disk
        add(p + "ffn_up.bias",       torch.zeros(4 * C))
        add(p + "ffn_down.weight",   sd[f"blocks.{i}.mlp.c_proj.weight"])  # [C,4C]->[4C,C] on-disk
        add(p + "ffn_down.bias",     torch.zeros(C))

    # final norm + tied output head -> [C, V] on-disk
    add("output_norm.weight", sd["ln_f.weight"])
    add("output_norm.bias",   sd["ln_f.bias"])
    add("output.weight",      sd["tok_emb.weight"])          # [V,C]->[C,V] on-disk

    # ---- tokenizer (gpt2 byte-level BPE) ----
    gw.add_tokenizer_model("gpt2")
    token_list = []
    for i in range(V):
        glyph = decoder.get(i, "")
        token_list.append(glyph.encode("utf-8"))
    gw.add_token_list(token_list)
    merge_strs = [f"{a} {b}" for a, b in merges]
    gw.add_token_merges([m.encode("utf-8") for m in merge_strs])
    gw.add_token_scores([0.0] * V)

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    f.close()
    print(f"\nWrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
