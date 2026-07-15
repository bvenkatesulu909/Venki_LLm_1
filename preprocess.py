"""Strip Gutenberg boilerplate / license headers & footers, concat -> corpus.txt.

Gutenberg files have a START/END marker around the real text. We keep only
the body between the first '*** START' and the last '*** END' (if present),
else fall back to skipping the first/last ~5% (headers/footers).
"""
import glob
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
out_path = os.path.join(DATA, "corpus.txt")


def clean(text):
    # find START/END markers
    start = None
    end = None
    low = text.lower()
    i = low.find("*** start of")
    if i != -1:
        nl = text.find("\n", i)
        start = nl + 1 if nl != -1 else i + 80
    j = low.rfind("*** end of")
    if j != -1:
        nl = text.find("\n", j)
        end = nl if nl != -1 else j + 20
    if start is not None and end is not None and end > start:
        return text[start:end]
    # fallback: drop first 5% and last 5%
    n = len(text)
    return text[int(n * 0.05):int(n * 0.95)]


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.txt")))
    files = [f for f in files if os.path.basename(f) != "corpus.txt"]
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                t = fh.read()
            body = clean(t)
            out.write(body)
            out.write("\n\n")
            total += len(body)
    print(f"corpus.txt written: {total:,} chars from {len(files)} files")


if __name__ == "__main__":
    main()
