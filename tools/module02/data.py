"""Extract the real data used by the module 2 figures (run once; writes data.json)."""
import os, sys, json, urllib.request
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regex, tiktoken
from bpe_core import *

D = {}

# 1. Sennrich et al. (2016) toy corpus, character-level BPE with an end-of-word marker
words = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
seqs = {w: list(w) + ["</w>"] for w in words}
steps = [{"merge": None, "state": {w: list(s) for w, s in seqs.items()}}]
for _ in range(7):
    pairs = Counter()
    for w, s in seqs.items():
        for a, b in zip(s, s[1:]):
            pairs[a, b] += words[w]
    (a, b), n = pairs.most_common(1)[0]
    for w in seqs:
        seqs[w] = merge_pair(seqs[w], (a, b), a + b)
    steps.append({"merge": [a, b, n], "state": {w: list(s) for w, s in seqs.items()}})
D["walk"] = {"words": words, "steps": steps}

# 2. parallel sentences: tokens needed by three real tokenizers
sents = {
    "English": "The weather is nice today, so let's go for a walk in the park.",
    "German": "Das Wetter ist heute schön, also lass uns im Park spazieren gehen.",
    "Spanish": "El tiempo es agradable hoy, así que vamos a pasear por el parque.",
    "Hindi": "आज मौसम अच्छा है, इसलिए चलो पार्क में टहलने चलते हैं।",
    "Japanese": "今日は天気がいいので、公園を散歩しましょう。",
}
encs = {n: tiktoken.get_encoding(n) for n in ("gpt2", "cl100k_base", "o200k_base")}
D["lang"] = {lang: {"chars": len(s), "bytes": len(s.encode()), **{n: len(e.encode(s)) for n, e in encs.items()}}
             for lang, s in sents.items()}

# 3. how real tokenizers split awkward strings
probe = [" hello", "hello", "Hello", "HELLO", "1234567", " 2024", "strawberry", "unbelievably"]
D["split"] = {n: {s: [e.decode_single_token_bytes(i).decode("utf-8", "replace") for i in e.encode(s)] for s in probe}
              for n, e in encs.items() if n != "o200k_base"}
D["probe"] = probe

# 4. train our BPE on TinyShakespeare, measure compression vs vocabulary size on held-out text
p = os.path.join(HERE, "..", "..", "data", "tinyshakespeare.txt")
if not os.path.exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", p)
text = open(p).read()
cut = int(len(text) * 0.9)
train, test = text[:cut], text[cut:]
ranks = train_bpe(train, 1024)
nbytes = len(test.encode())
sweep = []
for V in (256, 300, 384, 512, 640, 768, 1024):
    sub = {t: i for t, i in ranks.items() if i < V}
    sweep.append((V, nbytes / len(encode(test, sub))))
D["sweep"] = sweep
g2 = encs["gpt2"]
D["gpt2_bytes_per_token"] = nbytes / len(g2.encode(test))
inv = {i: t for t, i in ranks.items()}
D["first_merges"] = [inv[i].decode("utf-8", "replace") for i in range(256, 276)]
D["late_merges"] = [inv[i].decode("utf-8", "replace") for i in range(1000, 1010)]
json.dump(D, open(os.path.join(HERE, "data.json"), "w"), ensure_ascii=False)
print(json.dumps({k: D[k] for k in ("lang", "split", "sweep", "gpt2_bytes_per_token", "first_merges", "late_merges")}, ensure_ascii=False, indent=1)[:3500])
for s in D["walk"]["steps"]: print(s["merge"], s["state"])
