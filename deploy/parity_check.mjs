/* Node parity check: JS forward pass vs PyTorch _ref.json. Must match. */
import fs from 'fs';
import { Tokenizer } from './tokenizer.js';
import { buildModel, forward, sampleTop } from './app.js';

const blob = JSON.parse(fs.readFileSync(new URL('./model.json', import.meta.url)));
const tokObj = JSON.parse(fs.readFileSync(new URL('./tokenizer.json', import.meta.url)));
const ref = JSON.parse(fs.readFileSync(new URL('./_ref.json', import.meta.url)));
const cfg = blob.config;

const tok = new Tokenizer(tokObj);
const model = buildModel(blob);

// 1) tokenizer round-trip + ids match
const ids = tok.encode(ref.prompt);
const round = tok.decode(ids);
const idsMatch = JSON.stringify(ids) === JSON.stringify(ref.ids);
console.log('ids match Python:', idsMatch, ids);
console.log('round-trip equal:', round === ref.prompt);

// 2) forward pass on ref ids
const logits = forward(ids, model, cfg);
const Tt = ids.length - 1;
const vocab = cfg.vocab_size;
let maxDiff = 0, argmaxJs = 0, maxv = -Infinity;
for (let i = 0; i < vocab; i++) {
  const d = Math.abs(logits[Tt * vocab + i] - ref.last_logits[i]);
  if (d > maxDiff) maxDiff = d;
  if (logits[Tt * vocab + i] > maxv) { maxv = logits[Tt * vocab + i]; argmaxJs = i; }
}
console.log('argmax JS:', argmaxJs, ' argmax PyTorch:', ref.argmax, ' match:', argmaxJs === ref.argmax);
console.log('max abs logit diff:', maxDiff.toExponential(3));

// 3) full generation parity (deterministic greedy, temp tiny)
function genGreedy(promptIds, n) {
  let ctx = promptIds.slice();
  for (let s = 0; s < n; s++) {
    const lg = forward(ctx.slice(-cfg.block_size), model, cfg);
    const last = (ctx.length - Math.max(0, ctx.length - cfg.block_size));
    const row = lg.subarray((Math.min(ctx.length, cfg.block_size) - 1) * vocab);
    let best = 0, bv = -Infinity;
    for (let i = 0; i < vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
    ctx.push(best);
  }
  return ctx;
}
const g = genGreedy(ids, 20);
console.log('JS greedy continuation ids:', g.slice(ids.length));

const PARITY_OK = idsMatch && (round === ref.prompt) && (argmaxJs === ref.argmax) && (maxDiff < 2.0);
console.log('\n' + (PARITY_OK ? 'PARITY OK' : 'PARITY FAILED'));
process.exit(PARITY_OK ? 0 : 1);
