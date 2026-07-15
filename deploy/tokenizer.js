/* Venki_llm_1 — JS byte-level BPE tokenizer (port of bpe.py).
 * Loads tokenizer.json {encoder:{glyph->id}, merges:[[a,b],...]} and mirrors
 * the Python encode/decode exactly. */

function bytesToUnicode() {
  const bs = [];
  for (let b = '!'.charCodeAt(0); b <= '~'.charCodeAt(0); b++) bs.push(b);
  for (let b = '¡'.charCodeAt(0); b <= '¬'.charCodeAt(0); b++) bs.push(b);
  for (let b = '®'.charCodeAt(0); b <= 'ÿ'.charCodeAt(0); b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const m = {};
  for (let i = 0; i < bs.length; i++) m[bs[i]] = String.fromCodePoint(cs[i]);
  return m; // byte(0-255) -> glyph char
}

const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

class Tokenizer {
  constructor(obj) {
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = {};
    for (const k in this.byteEncoder) this.byteDecoder[this.byteEncoder[k]] = parseInt(k);
    this.encoder = obj.encoder;          // glyph -> id
    this.decoder = {};
    for (const g in this.encoder) this.decoder[this.encoder[g]] = g;
    this.bpeRanks = {};
    obj.merges.forEach((m, i) => { this.bpeRanks[m[0] + '\x01' + m[1]] = i; });
    this.eosId = obj.eos_id !== undefined ? obj.eos_id : -1;
  }

  _preTokenize(text) {
    const words = [];
    let m;
    PAT.lastIndex = 0;
    while ((m = PAT.exec(text)) !== null) {
      const token = m[0];
      const bytes = Array.from(new TextEncoder().encode(token)); // UTF-8 bytes
      const glyphs = bytes.map(b => this.byteEncoder[b]);
      words.push(glyphs);
    }
    return words;
  }

  _bpe(word) {
    if (word.length < 2) return word.slice();
    let pairs = this._getPairs(word);
    while (true) {
      let minRank = null, minPair = null;
      for (const p of pairs) {
        const r = this.bpeRanks[p[0] + '\x01' + p[1]];
        if (r !== undefined && (minRank === null || r < minRank)) { minRank = r; minPair = p; }
      }
      if (minPair === null) break;
      const a = minPair[0], b = minPair[1], ab = a + b;
      const newWord = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === a && word[i + 1] === b) { newWord.push(ab); i += 2; }
        else { newWord.push(word[i]); i++; }
      }
      word = newWord;
      if (word.length < 2) break;
      pairs = this._getPairs(word);
    }
    return word;
  }

  _getPairs(word) {
    const s = new Set();
    for (let i = 0; i < word.length - 1; i++) s.add([word[i], word[i + 1]]);
    return s;
  }

  encode(text) {
    const ids = [];
    for (const word of this._preTokenize(text)) {
      for (const piece of this._bpe(word)) ids.push(this.encoder[piece]);
    }
    return ids;
  }

  decode(ids) {
    let glyphStr = '';
    for (const id of ids) glyphStr += this.decoder[id];
    const bytes = [];
    for (const ch of glyphStr) bytes.push(this.byteDecoder[ch]);
    return new TextDecoder().decode(Uint8Array.from(bytes));
  }
}

if (typeof module !== 'undefined') module.exports = { Tokenizer };
