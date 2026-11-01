import regex as re
from collections import Counter

# GPT-2's pre-tokenization pattern: contractions, letters, digits, punctuation, whitespace
GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def merge_pair(toks, pair, new):
    """Replace every adjacent occurrence of `pair` in the token list with `new`."""
    out, i = [], 0
    while i < len(toks):
        if i < len(toks) - 1 and (toks[i], toks[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def train_bpe(text, vocab_size, pat=GPT2_PAT):
    """Learn byte-level BPE merges. Returns ranks: dict mapping token bytes -> id.
    Ids 0..255 are the single bytes; every merge adds one new id."""
    word_freq = Counter(pat.findall(text))                      # pre-tokenize, count unique pieces
    seqs = {w: [bytes([b]) for b in w.encode()] for w in word_freq}
    ranks = {bytes([i]): i for i in range(256)}
    while len(ranks) < vocab_size:
        pairs = Counter()
        for w, toks in seqs.items():
            for a, b in zip(toks, toks[1:]):
                pairs[a, b] += word_freq[w]
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]                     # most frequent adjacent pair
        new = a + b
        ranks[new] = len(ranks)
        for w in seqs:
            seqs[w] = merge_pair(seqs[w], (a, b), new)
    return ranks


def encode_piece(piece, ranks):
    """BPE-encode one pre-tokenized piece: repeatedly merge the adjacent pair
    whose concatenation has the lowest id (= was learned earliest)."""
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return [ranks[p] for p in parts]


def encode(text, ranks, pat=GPT2_PAT):
    return [i for piece in pat.findall(text) for i in encode_piece(piece.encode(), ranks)]


def decode(ids, ranks):
    inv = {i: t for t, i in ranks.items()}
    return b"".join(inv[i] for i in ids).decode("utf-8", errors="replace")
