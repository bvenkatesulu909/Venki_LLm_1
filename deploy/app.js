/* Venki_llm_1 — pure-JS GPT forward pass (2-bit weights).
 * Mirrors model.py exactly. Matmul convention: Y[B,T,C2] = X[B,T,C1] @ W[C1,C2]
 * (i.e. W columns are output dims). All weights stored as [out_dim, in_dim]
 * row-major to match PyTorch nn.Linear.weight shape. */

const GROUP = 64;

// ---- tiny linear-algebra helpers (flat Float32 arrays, row-major) ----
function matmul(X, W, B, n, inDim, outDim) {
  // X: [n*inDim], W: [outDim*inDim] (row-major, out first), B: [outDim]|null
  const Y = new Float32Array(n * outDim);
  for (let i = 0; i < n; i++) {
    const xr = i * inDim, yr = i * outDim;
    for (let o = 0; o < outDim; o++) {
      let s = B ? B[o] : 0;
      const wr = o * inDim;
      for (let k = 0; k < inDim; k++) s += X[xr + k] * W[wr + k];
      Y[yr + o] = s;
    }
  }
  return Y;
}

function dequant(w) {
  // w: {q, scale, min, shape, pad, group}
  const { q, scale, min, shape, pad, group } = w;
  const total = shape[0] * shape[1];
  const flat = new Float32Array(total);
  const nGroups = Math.ceil(total / group);
  const bytesPerGroup = group / 4; // 4 codes per byte
  let idx = 0;
  for (let g = 0; g < nGroups; g++) {
    const sc = scale[g], mn = min[g];
    const base = g * bytesPerGroup;
    for (let j = 0; j < group; j++) {
      if (idx >= total) break;
      const byte = q[base + (j >> 2)];
      const code = (byte >> ((j & 3) * 2)) & 3;
      flat[idx++] = code * sc + mn;
    }
  }
  // drop padding (global trailing codes)
  if (pad) return flat.subarray(0, total - pad);
  return flat;
}

function erf(x) {
  // Abramowitz & Stegun 7.1.26 approximation
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

function gelu(x) {
  // exact nn.GELU() = x * 0.5 * (1 + erf(x/sqrt(2)))
  return x * 0.5 * (1 + erf(x / Math.SQRT2));
}

function softmaxRow(r, start, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) max = Math.max(max, r[start + i]);
  let sum = 0;
  for (let i = 0; i < len; i++) { const e = Math.exp(r[start + i] - max); r[start + i] = e; sum += e; }
  for (let i = 0; i < len; i++) r[start + i] /= sum;
}

// LayerNorm (matches torch.nn.LayerNorm, eps=1e-5, elementwise_affine with weight+bias)
function layerNorm(x, w, b, n, dim) {
  const Y = new Float32Array(n * dim);
  const eps = 1e-5;
  for (let i = 0; i < n; i++) {
    const r = i * dim;
    let mean = 0;
    for (let j = 0; j < dim; j++) mean += x[r + j];
    mean /= dim;
    let variance = 0;
    for (let j = 0; j < dim; j++) { const d = x[r + j] - mean; variance += d * d; }
    variance /= dim;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let j = 0; j < dim; j++) Y[r + j] = (x[r + j] - mean) * inv * w[j] + b[j];
  }
  return Y;
}

// causal self-attention for one block: x [T, C] (n=B=1)
function attention(x, blk, cfg) {
  const T = x.length / cfg.n_embd;
  const C = cfg.n_embd, nh = cfg.n_head, hd = C / nh;
  const qkv = matmul(x, blk.attn_c_attn_w, blk.attn_c_attn_b, T, C, 3 * C); // [T, 3C]
  const out = new Float32Array(T * C);
  const scale = 1 / Math.sqrt(hd);
  for (let h = 0; h < nh; h++) {
    const off = h * hd;
    for (let t = 0; t < T; t++) {
      const scores = new Float32Array(T);
      for (let s = 0; s <= t; s++) {
        let dot = 0;
        for (let d = 0; d < hd; d++) {
          dot += qkv[t * 3 * C + off + d] * qkv[s * 3 * C + C + off + d];
        }
        scores[s] = dot * scale;
      }
      softmaxRow(scores, 0, t + 1);
      for (let d = 0; d < hd; d++) {
        let acc = 0;
        for (let s = 0; s <= t; s++) acc += scores[s] * qkv[s * 3 * C + 2 * C + off + d];
        out[t * C + off + d] = acc;
      }
    }
  }
  const Y = matmul(out, blk.attn_c_proj_w, blk.attn_c_proj_b, T, C, C);
  return Y;
}

function mlp(x, blk, cfg) {
  const T = x.length / cfg.n_embd, C = cfg.n_embd, I = 4 * C;
  const h = matmul(x, blk.mlp_c_fc_w, blk.mlp_c_fc_b, T, C, I);
  for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
  const Y = matmul(h, blk.mlp_c_proj_w, blk.mlp_c_proj_b, T, I, C);
  return Y;
}

function forward(idx, model, cfg) {
  const T = idx.length;
  const C = cfg.n_embd;
  // embeddings: tok_emb[idx] + pos_emb[:T]
  const x = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const tok = idx[t];
    for (let c = 0; c < C; c++) {
      x[t * C + c] = model.tok_emb[tok * C + c] + model.pos_emb[t * C + c];
    }
  }
  for (let li = 0; li < cfg.n_layer; li++) {
    const blk = model.blocks[li];
    const h1 = layerNorm(x, blk.ln1_w, blk.ln1_b, T, C);
    const a = attention(h1, blk, cfg);
    for (let i = 0; i < x.length; i++) x[i] += a[i];
    const h2 = layerNorm(x, blk.ln2_w, blk.ln2_b, T, C);
    const m = mlp(h2, blk, cfg);
    for (let i = 0; i < x.length; i++) x[i] += m[i];
  }
  const hf = layerNorm(x, model.ln_f_w, model.ln_f_b, T, C);
  // logits = hf @ head.weight (tied to tok_emb)  [T, vocab]
  const logits = matmul(hf, model.tok_emb, null, T, C, cfg.vocab_size);
  return logits; // [T, vocab]
}

// ---- sampling ----
function sampleTop(row, temperature, topK, topP) {
  // row: 1D Float32Array of logits for the last position [vocab]
  const vocab = row.length;
  const T = temperature || 1.0;
  const k = topK || 0;
  const p = topP || 0;
  const probs = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) probs[i] = row[i] / T;
  // top-k
  if (k > 0) {
    const arr = Array.from(probs).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const kth = arr[Math.min(k, vocab) - 1][0];
    for (let i = 0; i < vocab; i++) if (probs[i] < kth) probs[i] = -Infinity;
  }
  // top-p (nucleus)
  if (p > 0) {
    const order = Array.from(probs).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    let cum = 0; const keep = new Set();
    for (const [v, i] of order) { if (v === -Infinity) break; cum += Math.exp(v); keep.add(i); if (cum >= p) break; }
    for (let i = 0; i < vocab; i++) if (!keep.has(i)) probs[i] = -Infinity;
  }
  // softmax
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) max = Math.max(max, probs[i]);
  let sum = 0;
  for (let i = 0; i < vocab; i++) { const e = Math.exp(probs[i] - max); probs[i] = e; sum += e; }
  for (let i = 0; i < vocab; i++) probs[i] /= sum;
  // multinomial
  let r = Math.random();
  for (let i = 0; i < vocab; i++) { r -= probs[i]; if (r <= 0) return i; }
  return vocab - 1;
}

// ---- model loader (build flat arrays from exported JSON) ----
function buildModel(blob) {
  const cfg = blob.config;
  const W = blob.weights;
  const getF = (name) => Float32Array.from(W[name].f);
  const getQ = (name) => {
    const d = W[name];
    return { q: d.q, scale: d.scale, min: d.min, shape: d.shape, pad: d.pad, group: d.group };
  };
  const deqOrF = (name) => (W[name].f ? getF(name) : dequant(getQ(name)));

  const model = { cfg, tok_emb: deqOrF("tok_emb.weight") };
  model.pos_emb = getF("pos_emb");
  model.ln_f_w = getF("ln_f.weight");
  model.ln_f_b = getF("ln_f.bias");
  model.blocks = [];
  for (let l = 0; l < cfg.n_layer; l++) {
    const p = "blocks." + l + ".";
    model.blocks.push({
      ln1_w: getF(p + "ln1.weight"), ln1_b: getF(p + "ln1.bias"),
      ln2_w: getF(p + "ln2.weight"), ln2_b: getF(p + "ln2.bias"),
      // model built with bias=False -> no bias params; pass null
      attn_c_attn_w: deqOrF(p + "attn.c_attn.weight"), attn_c_attn_b: null,
      attn_c_proj_w: deqOrF(p + "attn.c_proj.weight"), attn_c_proj_b: null,
      mlp_c_fc_w: deqOrF(p + "mlp.c_fc.weight"), mlp_c_fc_b: null,
      mlp_c_proj_w: deqOrF(p + "mlp.c_proj.weight"), mlp_c_proj_b: null,
    });
  }
  return model;
}

// expose (browser + node)
if (typeof window !== 'undefined') window.VenkiLLM = { buildModel, forward, sampleTop };
if (typeof module !== 'undefined' && module.exports) module.exports = { buildModel, forward, sampleTop, layerNorm, attention, mlp, matmul, dequant, gelu };
