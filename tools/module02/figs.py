"""Build the SVG figures of docs/02_tokenization_bpe.html from real data (run data.py first).
Idempotent: figures are re-inserted after the <!--fig:name--> markers."""
import os, re, sys, json
from html import escape as E
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiktoken
from bpe_core import GPT2_PAT, encode_piece
D = json.load(open(os.path.join(HERE, "data.json")))
W = 480
NUM = {"gran": 1, "walk": 2, "merges": 3, "bytes": 4, "pipe": 5, "sweep": 6, "quirks": 7, "lang": 8}
ARROW = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'

def tokstr(b):
    try: return b.decode("utf-8")
    except UnicodeDecodeError: return "⟨" + " ".join(f"{x:02x}" for x in b) + "⟩"
def vis(s): return s.replace(" ", "·").replace("\n", "↵")
def t(x, y, s, cls="", anchor="start", size=12, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{cls}" {extra}>{E(s)}</text>'
def svg(h, body, label): return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(label)}">{body}</svg>'
def fig(key, s, cap): n = NUM[key]; return f'<figure class="fig" id="fig{n}">{s}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n'

def chips(items, x0, y0, wfun, fs=12, rowh=26, h=20, maxx=W - 8, gap=4, hl=False, tip=None):
    """Flow-layout chips; returns (svg, next_y)."""
    x, y, out = x0, y0, ""
    for it in items:
        w = wfun(it)
        if x + w > maxx and x > x0: x, y = x0, y + rowh
        out += f'<rect class="chip{" hl" if hl else ""}" x="{x}" y="{y}" width="{w:.0f}" height="{h}" rx="4"/>' + t(x + w / 2, y + h / 2 + fs * 0.35, vis(it), "mono", "middle", fs)
        x += w + gap
    return out, y + rowh

F = {}
g2 = tiktoken.get_encoding("gpt2")
def pieces(enc, s): return [enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in enc.encode(s)]

# 1 granularity
sent = "Unbelievably, tokenization isn't trivial."
rows = [("characters", list(sent)), ("BPE tokens (GPT-2)", pieces(g2, sent)), ("words", sent.split(" "))]
body, y = "", 6
for name, items in rows:
    body += t(8, y + 12, f"{name}: {len(items)}", "", "start", 12, 'font-weight="600"'); y += 18
    c, y = chips(items, 8, y, (lambda it: 13) if name == "characters" else (lambda it: 14 + 7.5 * len(vis(it))), fs=11 if name == "characters" else 12, rowh=26, h=20, gap=3 if name == "characters" else 4)
    body += c; y += 6
F["gran"] = fig("gran", svg(y, body, "The same sentence as characters, BPE tokens and words"),
    f"<code>{E(sent)}</code> at three granularities (real GPT-2 output). Characters give long sequences; words give short ones, but no vocabulary of whole words is ever complete; BPE sits in between, splitting the rare word <i>Unbelievably</i> into reusable pieces. <code>·</code> marks a leading space.")

# 2 walkthrough
steps = D["walk"]["steps"]; order = list(D["walk"]["words"]); body = ""
cw, chh = 236, 92
for k, st in enumerate(steps):
    cx, cy = 4 + (k % 2) * (cw + 4), 4 + (k // 2) * chh
    new = (st["merge"][0] + st["merge"][1]) if st["merge"] else None
    head = "start: characters + end marker" if k == 0 else f"merge {k}: {st['merge'][0]} + {st['merge'][1]}  ({st['merge'][2]}×)"
    body += f'<rect class="box" x="{cx}" y="{cy}" width="{cw}" height="{chh-6}" rx="6" style="stroke:var(--bd)"/>' + t(cx + 8, cy + 15, head, "", "start", 11, 'font-weight="600"')
    for j, w in enumerate(order):
        spans = "".join(f'<tspan style="font-weight:700;fill:var(--s2)">{E(s)} </tspan>' if s == new else f"<tspan>{E(s)} </tspan>" for s in st["state"][w])
        body += f'<text x="{cx+8}" y="{cy+32+j*15}" font-size="11" class="mono"><tspan class="mut" style="fill:var(--mut)">{w:<7}</tspan>{spans}</text>'.replace("<tspan class", '<tspan xml:space="preserve" class', 1)
F["walk"] = fig("walk", svg(4 + 4 * chh, body, "Byte-pair merges on the toy corpus low, lower, newest, widest"),
    "BPE training on the toy corpus of Sennrich et al. (2016): <code>low</code> ×5, <code>lower</code> ×2, <code>newest</code> ×6, <code>widest</code> ×3. Each panel shows the words after one more merge; the symbol just created is highlighted and the header gives the merged pair and how often it occurred. Ties are broken by first occurrence. Real output of our implementation.")

# 3 learned merges
body, y = t(8, 16, "first merges learned from Shakespeare (most frequent pairs)", "", "start", 12, 'font-weight="600"'), 24
c, y = chips(D["first_merges"], 8, y, lambda it: 14 + 8 * len(vis(it)), fs=12); body += c
body += t(8, y + 14, "merges near the end of the 1,024-token vocabulary (rare fragments)", "", "start", 12, 'font-weight="600"'); y += 22
c, y = chips(D["late_merges"], 8, y, lambda it: 14 + 8 * len(vis(it)), fs=12); body += c
F["merges"] = fig("merges", svg(y, body, "Early and late merges learned by BPE on Shakespeare"),
    "Real merges from our BPE trained on TinyShakespeare. Early merges are the most frequent letter pairs and short words (<code>·the</code>), late ones are rare fragments and names. <code>·</code> marks a leading space. The vocabulary is a frequency-ordered record of the training text.")

# 3b characters -> UTF-8 bytes -> GPT-2 base ids
cols_x, body, x = [], "", 8
body += t(344, 20, "character", "mut", "start", 11) + t(344, 62, "UTF-8 bytes", "mut", "start", 11) + t(344, 104, "GPT-2 base ids", "mut", "start", 11)
for ch in ["A", "é", "世", "😀"]:
    bs = list(ch.encode("utf-8")); w = 28 * len(bs) - 2
    body += f'<rect class="chip hl" x="{x}" y="4" width="{w}" height="24" rx="4"/>' + t(x + w / 2, 22, ch, "", "middle", 15)
    for k, b in enumerate(bs):
        bx_ = x + 28 * k; i = g2.encode_single_token(bytes([b]))
        body += f'<path d="M{x + w / 2:.0f} 30L{bx_ + 13} 42" class="ln"/>'
        body += f'<rect class="chip" x="{bx_}" y="44" width="26" height="20" rx="4"/>' + t(bx_ + 13, 58, str(b), "mono", "middle", 11)
        body += f'<path d="M{bx_ + 13} 66V82" class="ln" marker-end="url(#ah)"/>'
        body += f'<rect class="chip hl" x="{bx_}" y="84" width="26" height="20" rx="4"><title>byte {b} (hex {b:02x}) is GPT-2 token id {i}</title></rect>' + t(bx_ + 13, 98, str(i), "mono", "middle", 11)
    x += 28 * len(bs) + 14
body += t(8, 128, "Characters to bytes: the fixed UTF-8 standard.", "mut", "start", 10.5) + t(8, 142, "Bytes to ids: a fixed, arbitrary lookup table. Nothing is learned.", "mut", "start", 10.5)
F["bytes"] = fig("bytes", svg(150, ARROW + body, "Characters to UTF-8 bytes to GPT-2 base token ids"),
    "Four characters written as UTF-8 bytes (decimal), then looked up as GPT-2's 256 base tokens. <code>A</code> is one byte (65), <code>é</code> two, <code>世</code> three, the emoji four. The ids in the bottom row (A is 32, not 65) come from GPT-2's ordering of the 256 bytes, which is a labelling choice; the model only ever sees them as rows of an embedding table. BPE merges are then learned on top of these base tokens. Real values read from <code>tiktoken</code>.")

# 4 pipeline
ex = "Hello, 世界!"
rk = g2._mergeable_ranks
pcs = GPT2_PAT.findall(ex)
toks = [rk_i for p in pcs for rk_i in encode_piece(p.encode(), rk)]
inv = {i: b for b, i in rk.items()}
bytes_of = " ".join(f"{b:02x}" for b in pcs[2].encode())
stages = [("text", f"“{ex}”"),
          ("pre-tokenize (regex)", "[" + ", ".join(repr(p) for p in pcs) + "]"),
          ("each piece → bytes", f"{pcs[2]!r} → {bytes_of}"),
          ("apply merges", " | ".join(tokstr(inv[i]) if tokstr(inv[i]).startswith("⟨") else vis(tokstr(inv[i])) for i in toks)),
          ("token ids", str(toks))]
body = ARROW
for k, (a, b) in enumerate(stages):
    y = 6 + k * 52
    body += f'<rect class="box" x="6" y="{y}" width="150" height="38" rx="8"/>' + t(81, y + 24, a, "", "middle", 11.5)
    body += t(166, y + 24, b, "mono", "start", 11)
    if k < len(stages) - 1: body += f'<path d="M81 {y+38}V{y+52}" class="ln" marker-end="url(#ah)"/>'
F["pipe"] = fig("pipe", svg(6 + len(stages) * 52, body, "Byte-level BPE pipeline from text to token ids"),
    "The GPT-2 tokenization pipeline on a real example. The regex splits the text first (note the leading space stays with <code>·世界</code>), merges happen only inside each piece, and a character with no merge in the vocabulary is simply left as raw bytes: here 世 and 界 are not single tokens in GPT-2, so they appear as byte fragments (shown in hex, such as <code>⟨e4 b8⟩</code>). Output is real GPT-2 (<code>tiktoken</code>).")

# 5 sweep
sw = D["sweep"]; ox, oy, pw, ph = 46, 14, 410, 160; ymin, ymax = 0.5, 3.4
X = lambda v: ox + (v - 256) / (1024 - 256) * pw; Y = lambda v: oy + ph - (v - ymin) / (ymax - ymin) * ph
body = f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" class="axis"/>'
for g in (1, 2, 3):
    body += f'<line x1="{ox}" y1="{Y(g):.1f}" x2="{ox+pw}" y2="{Y(g):.1f}" class="grid"/>' + t(ox - 6, Y(g) + 4, str(g), "mut", "end", 10)
body += f'<line x1="{ox}" y1="{Y(D["gpt2_bytes_per_token"]):.1f}" x2="{ox+pw}" y2="{Y(D["gpt2_bytes_per_token"]):.1f}" stroke="var(--s2)" stroke-width="2" stroke-dasharray="5 4"/>'
body += t(ox + pw, Y(D["gpt2_bytes_per_token"]) - 6, f"GPT-2 tokenizer (50,257 tokens): {D['gpt2_bytes_per_token']:.2f}", "", "end", 11)
body += f'<polyline fill="none" stroke="var(--s1)" stroke-width="2.5" points="{" ".join(f"{X(v):.1f},{Y(b):.1f}" for v, b in sw)}"/>'
for v, b in sw:
    body += f'<circle cx="{X(v):.1f}" cy="{Y(b):.1f}" r="4" fill="var(--s1)" stroke="var(--bg)" stroke-width="2"><title>vocab {v}: {b:.2f} bytes/token</title></circle>' + (t(X(v), oy + ph + 14, str(v), "mut", "middle", 10) if v in (256, 512, 768, 1024) else "") + (t(X(v) + 8, Y(b) + 16, f"{b:.2f}", "mut", "start", 10) if v == 256 else t(X(v), Y(b) - 9, f"{b:.2f}", "mut", "middle", 10))
body += t(ox + pw / 2, oy + ph + 30, "vocabulary size (merges learned)", "mut", "middle", 11) + t(ox, oy - 3, "bytes per token (higher = shorter sequences)", "mut", "start", 10.5)
F["sweep"] = fig("sweep", svg(oy + ph + 38, body, "Bytes per token as a function of vocabulary size"),
    "Compression of held-out Shakespeare as our BPE learns more merges (blue; one point per vocabulary size). Returns diminish quickly. The dashed line is GPT-2's 50,257-token tokenizer on the same text, for scale. Real measurements from the notebook.")

# 6 quirks
probe = [" hello", "Hello", "HELLO", "1234567", " 2024", "strawberry"]
body, y = t(8, 12, "string", "mut", "start", 11) + t(108, 12, "tokenizer → pieces", "mut", "start", 11), 22
for s in probe:
    body += t(8, y + 28, vis(s), "mono", "start", 12, 'font-weight="600"')
    for k, (nm, key) in enumerate((("GPT-2", "gpt2"), ("GPT-4", "cl100k_base"))):
        body += t(108, y + 14 + k * 25, nm, "mut", "start", 10)
        c, _ = chips(D["split"][key][s], 148, y + k * 25, lambda it: 12 + 7.6 * len(vis(it)), fs=11.5, rowh=25, h=20, gap=3); body += c
    y += 56
F["quirks"] = fig("quirks", svg(y, body, "How GPT-2 and GPT-4 tokenizers split awkward strings"),
    "Real splits of awkward strings. A leading space or a change of case produces different tokens; <code>HELLO</code> shatters; <code>1234567</code> is chopped differently by each tokenizer (GPT-4 groups digits in threes); and <i>strawberry</i> is never seen as letters. <code>·</code> marks a space.")

# 7 languages
L = D["lang"]; names = ["gpt2", "cl100k_base", "o200k_base"]; lab = ["GPT-2", "GPT-4", "GPT-4o"]; cols = ["var(--s1)", "var(--s2)", "var(--s3)"]
mx = max(v[n] for v in L.values() for n in names); bx = 84; bw = 330; body = ""
for k, (nm, c) in enumerate(zip(lab, cols)):
    body += f'<rect x="{8+k*92}" y="6" width="12" height="12" rx="2" fill="{c}"/>' + t(26 + k * 92, 17, nm, "", "start", 11.5)
y = 30
for lang, v in L.items():
    body += t(bx - 8, y + 24, lang, "", "end", 12)
    for k, n in enumerate(names):
        w = v[n] / mx * bw
        body += f'<rect x="{bx}" y="{y+k*14}" width="{w:.1f}" height="11" rx="3" fill="{cols[k]}"><title>{lang}, {lab[k]}: {v[n]} tokens</title></rect>' + t(bx + w + 5, y + k * 14 + 10, str(v[n]), "mut", "start", 10.5)
    y += 52
F["lang"] = fig("lang", svg(y, body, "Tokens needed for one sentence in five languages under three tokenizers"),
    "Tokens needed for one sentence of roughly the same meaning (“the weather is nice today, let's go for a walk in the park”) in five languages, under three real tokenizers. The Hindi sentence costs about five times the English one under GPT-2; GPT-4o's newer, more multilingual vocabulary closes the gap almost entirely for this sentence. One sentence per language is an illustration, not a benchmark.")

# ---- patch the lesson
p = os.path.join(HERE, "..", "..", "docs", "02_tokenization_bpe.html"); h = open(p).read()
h = re.sub(r'\n?<figure class="fig" id="fig\d+">.*?</figure>\n', "", h, flags=re.S)
for key, fg in F.items():
    m = f"<!--fig:{key}-->"; assert h.count(m) == 1, key
    h = h.replace(m, m + "\n" + fg)
open(p, "w").write(h); print("figures:", h.count("<figure"))
