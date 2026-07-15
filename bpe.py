"""Venki_llm_1 — from-scratch byte-level BPE tokenizer.

No tiktoken / transformers shortcuts. Implements the GPT-2 byte->unicode
glyph scheme, GPT-2 pre-tokenization regex, a frequency-map merge learner
(NOT stride-1 over the corpus), and reversible encode/decode with a
round-trip assert.
"""
import json
import regex as re
from collections import Counter

# GPT-2 byte-to-unicode mapping (keeps UTF-8 reversible, printable glyphs).
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, vocab_size=4000):
        self.vocab_size = vocab_size
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = {}        # token string -> id
        self.decoder = {}        # id -> token string
        self.merges = {}         # (a,b) -> id (rank implicit by insertion)
        self.bpe_ranks = {}      # (a,b) -> rank

    # ---- training ----
    def train(self, text, max_chars=200000):
        # Train on a representative subsample: full-corpus BPE is O(vocab x corpus)
        # in pure Python and the merges learned from ~200KB generalize well.
        if len(text) > max_chars:
            text = text[:max_chars]
        # word = list of glyphs (one per byte of the UTF-8 bytes of the pretoken)
        words = self._pre_tokenize(text)
        # seed vocab with ALL 256 byte-glyphs so every byte is always encodable
        vocab = set(self.byte_encoder.values())
        for w in words:
            vocab.update(w)
        # initial merges: none; tokens are single glyphs
        self.encoder = {g: i for i, g in enumerate(sorted(vocab))}
        # add special? none for now
        # frequency map of words -> count
        word_counts = Counter(tuple(w) for w in words)
        # iterative merges
        next_id = len(self.encoder)
        merges = []
        while len(self.encoder) < self.vocab_size:
            # count pairs over word_counts (skip single-glyph words: no pair)
            pairs = Counter()
            for word, c in word_counts.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += c
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: kv[1])[0]
            merges.append(best)
            # apply merge
            a, b = best
            ab = a + b
            new_counts = Counter()
            for word, c in word_counts.items():
                # skip words that cannot contain the pair (quick filter)
                if len(word) < 2 or a not in word:
                    new_counts[word] += c
                    continue
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(ab)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_counts[tuple(new_word)] += c
            word_counts = new_counts
            if ab not in self.encoder:
                self.encoder[ab] = next_id
                next_id += 1
            if len(self.encoder) >= self.vocab_size:
                break
        self.merges = merges
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.decoder = {v: k for k, v in self.encoder.items()}
        return len(self.encoder)

    def _pre_tokenize(self, text):
        # returns list of words, each word a tuple of glyphs (per-byte unicode)
        out = []
        for match in PAT.finditer(text):
            token = match.group(0)
            # convert to bytes, then to glyphs
            b = token.encode("utf-8")
            out.append(tuple(self.byte_encoder[x] for x in b))
        return out

    def _get_pairs(self, word):
        return set(zip(word[:-1], word[1:]))

    def _bpe(self, token_tuple):
        word = list(token_tuple)
        if len(word) < 2:
            return word
        pairs = self._get_pairs(word)
        while True:
            min_rank = None
            min_pair = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is not None and (min_rank is None or r < min_rank):
                    min_rank = r
                    min_pair = p
            if min_pair is None:
                break
            a, b = min_pair
            ab = a + b
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(ab)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) < 2:
                break
            pairs = self._get_pairs(word)
        return word

    # ---- encode / decode ----
    def encode(self, text):
        ids = []
        for word in self._pre_tokenize(text):
            for piece in self._bpe(word):
                ids.append(self.encoder[piece])
        return ids

    def decode(self, ids):
        # ids -> glyphs -> bytes -> utf-8 text
        text_chars = "".join(self.decoder[i] for i in ids)
        # convert glyphs back to bytes
        out_bytes = bytearray()
        for ch in text_chars:
            out_bytes.append(self.byte_decoder[ch])
        return out_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        obj = {
            "vocab_size": self.vocab_size,
            "encoder": self.encoder,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        t = cls(obj["vocab_size"])
        t.encoder = obj["encoder"]
        t.merges = [tuple(m) for m in obj["merges"]]
        t.bpe_ranks = {tuple(m): i for i, m in enumerate(t.merges)}
        t.decoder = {v: k for k, v in t.encoder.items()}
        return t


if __name__ == "__main__":
    # round-trip self-test
    t = Tokenizer(vocab_size=600)
    import sys
    data = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else "Hello world, this is Venki_llm_1!"
    t.train(data)
    s = "The quick brown fox jumps over the lazy dog. Venki_llm_1 — émojis ✨ and numbers 12345 work too."
    ids = t.encode(s)
    dec = t.decode(ids)
    assert dec == s, f"roundtrip failed: {dec!r} != {s!r}"
    print(f"vocab={len(t.encoder)} roundtrip OK, sample ids={ids[:20]}")
