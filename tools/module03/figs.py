"""Build the SVG figures of docs/03_embeddings_bigram.html from real data (run data.py first).
Idempotent: figures are re-inserted after the <!--fig:name--> markers."""
import os, re, json, math
from html import escape as E
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))
W = 480
NUM = {"lookup": 1, "arch": 2, "heat": 3, "gd": 4, "sweep": 5, "geo": 6}
ARROW = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'

def t(x, y, s, cls="", anchor="start", size=12, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{cls}" {extra}>{E(s)}</text>'
def svg(h, body, label): return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(label)}">{body}</svg>'
def fig(key, s, cap): n = NUM[key]; return f'<figure class="fig" id="fig{n}">{s}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n'
def vis(s): return s.replace(" ", "·").replace("\n", "↵")
def arrow(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#ah)"/>'
def axes(ox, oy, pw, ph): return f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" class="axis"/>'
F = {}

# 1 lookup, real GPT-2 rows
body = ARROW + t(8, 16, "token", "mut", "start", 11) + t(120, 16, "id", "mut", "start", 11) + t(208, 16, "row of the table (first 5 of 768 numbers)", "mut", "start", 11)
for k, r in enumerate(D["gpt2"]["lookup"]):
    y = 26 + k * 44
    body += f'<rect class="chip hl" x="8" y="{y}" width="70" height="30" rx="5"/>' + t(43, y + 20, vis(r["tok"]), "mono", "middle", 13)
    body += arrow(80, y + 15, 112, y + 15) + t(120, y + 20, str(r["id"]), "mono", "start", 13) + arrow(168, y + 15, 200, y + 15)
    body += f'<rect class="box" x="204" y="{y}" width="270" height="30" rx="5" style="stroke:var(--s1)"/>'
    body += t(212, y + 20, "  ".join(f"{v:+.3f}" for v in r["row"]) + "  …", "mono", "start", 11.5)
body += t(8, 168, "wte has 50,257 rows × 768 columns = 38,597,376 numbers; GPT-2 small, real values", "mut", "start", 11)
F["lookup"] = fig("lookup", svg(178, body, "Three GPT-2 tokens, their ids and the first five numbers of their embedding rows"),
    "An embedding layer is a table lookup. The tokenizer turns <code>The cat sat</code> into the ids 464, 3797, 3332 (note the leading space inside <code>␣cat</code>), and each id selects one row of GPT-2's token-embedding matrix <code>wte</code>. Values are the real weights of GPT-2 small; only the first five of 768 numbers per row are shown.")

# 2 architecture of the Bengio-style neural LM (schematic)
body = ARROW; ys = [30, 70, 110]
for k, y in enumerate(ys):
    lab = f"x[t−{3-k}]"
    body += f'<rect class="chip" x="6" y="{y}" width="74" height="30" rx="5"/>' + t(43, y + 20, lab, "mono", "middle", 12)
    body += arrow(82, y + 15, 112, y + 15)
    body += f'<rect x="114" y="{y}" width="58" height="30" rx="4" fill="var(--s1)"/>' + t(143, y + 20, "C[x]", "mono", "middle", 12, 'style="fill:#fff"')
    body += arrow(174, y + 15, 204, y + 15 if True else y)
body += t(143, 20, "same table C", "mut", "middle", 11) + t(143, 156, "d numbers each", "mut", "middle", 11)
body += f'<rect class="box" x="206" y="26" width="58" height="118" rx="5" style="stroke:var(--s1)"/>'
for k, y in enumerate(ys): body += f'<rect x="212" y="{32+k*38}" width="46" height="30" rx="3" fill="var(--s1)" opacity="{0.9-0.15*k}"/>'
body += t(235, 20, "concatenate", "mut", "middle", 11) + t(235, 156, "n·d numbers", "mut", "middle", 11)
body += arrow(266, 85, 292, 85) + f'<rect class="box" x="294" y="26" width="66" height="118" rx="5" style="stroke:var(--s2)"/>' + t(327, 80, "tanh", "", "middle", 13, 'font-weight="600"') + t(327, 98, "hidden", "", "middle", 12) + t(327, 20, "h units", "mut", "middle", 11)
body += arrow(362, 85, 386, 85) + f'<rect class="box" x="388" y="46" width="86" height="78" rx="5" style="stroke:var(--s3)"/>' + t(431, 78, "logits", "", "middle", 12, 'font-weight="600"') + t(431, 94, "then softmax", "", "middle", 11.5) + t(431, 40, "V outputs", "mut", "middle", 11) + t(431, 142, "P(next token)", "mut", "middle", 11)
F["arch"] = fig("arch", svg(168, body, "Architecture of a Bengio-style neural language model with a three-token context"),
    "The neural language model of Bengio et al. (2003), drawn for a context of n=3 tokens. Each previous token indexes the <i>same</i> embedding table C, the looked-up vectors are concatenated, one tanh hidden layer mixes them, and a softmax over the V tokens gives the next-token distribution. Schematic, not measured data (the original paper also adds optional direct input-to-output connections, omitted here).")

# 3 bigram heatmap (top 30 characters by frequency)
chars = D["meta"]["chars"]; P = D["bigram"]["P"]; fr = D["bigram"]["freq"]
top = sorted(range(len(chars)), key=lambda i: -fr[i])[:30]
cs, ox, oy = 12, 38, 40
body = t(ox + 30 * cs / 2, 12, "next character", "mut", "middle", 11) + t(10, oy + 30 * cs / 2, "previous", "mut", "middle", 11, f'transform="rotate(-90 10 {oy+30*cs/2})"')
for j, c in enumerate(top):
    body += t(ox + j * cs + cs / 2, oy - 4, vis(chars[c]), "mono", "middle", 11)
    body += t(ox - 4, oy + j * cs + cs / 2 + 4, vis(chars[c]), "mono", "end", 11)
for i, a in enumerate(top):
    for j, b in enumerate(top):
        p = P[a][b]
        if p < 0.002: continue
        body += f'<rect class="cell" x="{ox+j*cs}" y="{oy+i*cs}" width="{cs-1}" height="{cs-1}" opacity="{min(1, (p/0.5)**0.5):.2f}"><title>after {vis(chars[a])!r}: {vis(chars[b])!r} with probability {p:.3f}</title></rect>'
cov = sum(fr[i] for i in top)
body += t(ox + 30 * cs + 8, oy + 12, "darker =", "mut", "start", 11) + t(ox + 30 * cs + 8, oy + 26, "likelier", "mut", "start", 11)
F["heat"] = fig("heat", svg(oy + 30 * cs + 8, body, "Bigram probability table for the 30 most frequent characters of Shakespeare"),
    f"The bigram table P(next | previous) estimated from {D['meta']['n_train']:,} training characters of Shakespeare, restricted to the 30 most frequent characters (they make up {cov*100:.0f}% of the text); each row sums to 1 over all {D['meta']['V']} characters. Cell darkness is the probability (capped at 0.5; hover for exact values). <code>·</code> is a space and <code>↵</code> a newline. Real counts.")

# 4 gradient descent on the table vs the counts
B = D["bigram"]; ox, oy, pw, ph = 46, 16, 410, 170; ymin, ymax = 2.2, 4.4
X = lambda s: ox + s / 300 * pw; Y = lambda v: oy + ph - (v - ymin) / (ymax - ymin) * ph
body = axes(ox, oy, pw, ph)
for g in (2.5, 3.0, 3.5, 4.0): body += f'<line x1="{ox}" y1="{Y(g):.1f}" x2="{ox+pw}" y2="{Y(g):.1f}" class="grid"/>' + t(ox - 6, Y(g) + 4, f"{g:.1f}", "mut", "end", 11)
body += f'<line x1="{ox}" y1="{Y(B["mle_train"]):.1f}" x2="{ox+pw}" y2="{Y(B["mle_train"]):.1f}" stroke="var(--s2)" stroke-width="2" stroke-dasharray="5 4"/>'
body += t(ox + pw, Y(B["mle_train"]) + 16, f"closed form from counts: {B['mle_train']:.3f}", "", "end", 11.5)
body += f'<polyline fill="none" stroke="var(--s1)" stroke-width="2.5" points="{" ".join(f"{X(s):.1f},{Y(l):.1f}" for s, l in B["curve"])}"/>'
for s, l in B["curve"]:
    body += f'<circle cx="{X(s):.1f}" cy="{Y(l):.1f}" r="3.5" fill="var(--s1)" stroke="var(--bg)" stroke-width="1.5"><title>step {s}: loss {l:.3f}</title></circle>'
body += t(X(0) + 8, Y(B["curve"][0][1]) + 4, f"start: ln 65 = {D['meta']['lnV']:.3f}", "", "start", 11.5)
body += t(X(30) + 10, Y(2.815) - 8, "gradient descent on the 65×65 table", "", "start", 11.5, 'style="fill:var(--s1)"')
for s in (0, 100, 200, 300): body += t(X(s), oy + ph + 15, str(s), "mut", "middle", 11)
body += t(ox + pw / 2, oy + ph + 31, "Adam steps (full batch)", "mut", "middle", 11) + t(ox, oy - 4, "cross-entropy on training text (nats per character)", "mut", "start", 11)
F["gd"] = fig("gd", svg(oy + ph + 38, body, "Training a bigram table by gradient descent converges to the count-based solution"),
    f"Fitting the bigram table by gradient descent on the cross-entropy loss. It starts at ln 65 (a uniform guess over 65 characters) and approaches the loss of the closed-form answer, which is simply the normalised pair counts (dashed). After 300 steps the gap is {B['gd_train']-B['mle_train']:.4f} nats. Real run.")

# 5 context-length sweep
S = D["sweep"]; ox, oy, pw, ph = 46, 30, 330, 170; ymin, ymax = 0.8, 3.5
X = lambda n: ox + n / 8 * pw; Y = lambda v: oy + ph - (v - ymin) / (ymax - ymin) * ph
body = axes(ox, oy, pw, ph)
for g in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5): body += f'<line x1="{ox}" y1="{Y(g):.1f}" x2="{ox+pw}" y2="{Y(g):.1f}" class="grid"/>' + t(ox - 6, Y(g) + 4, f"{g:.1f}", "mut", "end", 11)
def series(key, col, dash, lab, ns):
    pts = [(r["n"], r[key]) for r in S if key in r]
    s = f'<polyline fill="none" stroke="{col}" stroke-width="2.5" {"stroke-dasharray=" + chr(34) + "5 4" + chr(34) if dash else ""} points="{" ".join(f"{X(n):.1f},{Y(v):.1f}" for n, v in pts)}"/>'
    for n, v in pts: s += f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}" stroke="var(--bg)" stroke-width="1.5"><title>{lab}, context {n}: {v:.3f} nats/char</title></circle>'
    return s
body += series("ng_test", "var(--s1)", False, "count n-gram, held-out", 7) + series("ng_train", "var(--s1)", True, "count n-gram, training", 7)
body += series("mlp_test", "var(--s2)", False, "neural LM, held-out", 8) + series("mlp_train", "var(--s2)", True, "neural LM, training", 8)
last = lambda k: [r for r in S if k in r][-1]
body += t(X(6) + 8, Y(last("ng_test")["ng_test"]) + 4, "n-gram, held-out", "", "start", 11.5, 'style="fill:var(--s1)"')
body += t(X(6) + 8, Y(last("ng_train")["ng_train"]) + 4, "n-gram, training", "", "start", 11.5, 'style="fill:var(--s1)"')
body += t(X(8) + 8, Y(S[-1]["mlp_test"]) + 4, "neural, held-out", "", "start", 11.5, 'style="fill:var(--s2)"')
body += t(X(8) + 8, Y(S[-1]["mlp_train"]) + 16, "neural, training", "", "start", 11.5, 'style="fill:var(--s2)"')
for n in range(9): body += t(X(n), oy + ph + 15, str(n), "mut", "middle", 11)
body += t(ox + pw / 2, oy + ph + 31, "context length n (previous characters)", "mut", "middle", 11) + t(ox, oy - 14, "cross-entropy (nats per character); solid = held-out text, dashed = training text", "mut", "start", 11)
F["sweep"] = fig("sweep", svg(oy + ph + 38, body, "Held-out and training loss versus context length for count n-grams and a neural language model"),
    "Predicting the next character from the previous n, on TinyShakespeare. <b>Blue:</b> a count-based n-gram with add-α smoothing (α picked on a validation slice). <b>Orange:</b> a small neural LM in the style of Bengio et al. (16-dim embeddings, one 256-unit tanh layer, 6,000 Adam steps). The counting model improves until n=4, then memorises its training text (dashed line keeps falling) and gets worse on new text, because most long contexts were never seen. The neural model stops improving after n≈4 but does not collapse. Real runs, one seed.")

# 6 GPT-2 geometry
Gp = D["gpt2"]["groups"]["points"]; ve = D["gpt2"]["groups"]["var_explained"]
ox, oy, pw, ph = 12, 26, 456, 270
xs = [p[2] for p in Gp]; ys = [p[3] for p in Gp]; x0, x1, y0, y1 = min(xs) - 0.55, max(xs) + 0.5, min(ys) - 0.25, max(ys) + 0.3
X = lambda v: ox + (v - x0) / (x1 - x0) * pw; Y = lambda v: oy + ph - (v - y0) / (y1 - y0) * ph
col = {"countries": "var(--s1)", "capitals": "var(--s2)"}
body = f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" class="axis"/>'
by = {(p[0], p[1]): (p[2], p[3]) for p in Gp}
pairs = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo")]
for c, k in pairs:
    a, b2 = by[("countries", c)], by[("capitals", k)]
    body += f'<line x1="{X(a[0]):.1f}" y1="{Y(a[1]):.1f}" x2="{X(b2[0]):.1f}" y2="{Y(b2[1]):.1f}" stroke="var(--mut)" stroke-width="1" stroke-dasharray="3 3"/>'
place = {"France": "below", "Madrid": "above"}   # default: countries label left, capitals right
for g, name, x, y in Gp:
    body += f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{col[g]}" stroke="var(--bg)" stroke-width="1.5"><title>{name} ({g})</title></circle>'
    pl = place.get(name, "left" if g == "countries" else "right")
    if pl == "left": body += t(X(x) - 9, Y(y) + 4, name, "", "end", 11.5)
    elif pl == "right": body += t(X(x) + 9, Y(y) + 4, name, "", "start", 11.5)
    elif pl == "above": body += t(X(x), Y(y) - 10, name, "", "middle", 11.5)
    else: body += t(X(x), Y(y) + 19, name, "", "middle", 11.5)
body += t(X(-1.75), Y(1.85), "countries", "", "start", 12.5, 'font-weight="700" style="fill:var(--s1)"') + t(X(1.9), Y(-1.75), "capitals", "", "end", 12.5, 'font-weight="700" style="fill:var(--s2)"')
body += t(ox + pw / 2, oy + ph + 16, f"first principal axis ({ve[0]*100:.0f}% of variance)", "mut", "middle", 11) + t(ox, oy - 8, f"second principal axis ({ve[1]*100:.0f}%)", "mut", "start", 11)
F["geo"] = fig("geo", svg(oy + ph + 24, body, "Two-dimensional projection of GPT-2 embeddings for eight countries and their capitals"),
    f"GPT-2's token embeddings for eight countries and their capitals, projected to 2-D by PCA (fit on just these 16 points, so the axes show whatever varies most among them). Countries sit on the left and capitals on the right, and the dashed country-to-capital connectors are almost parallel: in this plane, &ldquo;capital of&rdquo; is roughly one shared offset. Real values from <code>wte</code> of GPT-2 small. The two axes keep only about {sum(ve)*100:.0f}% of the variance, so distances in the picture are a rough guide.")

# ---- generated text regions: <!--gen:name-->...<!--/gen:name-->
G = {}
G["sample_bigram"] = E(D["bigram"]["sample"].lstrip("\n"))
G["sample_mlp"] = E(D["mlp_sample"].lstrip("\n"))
G["nn"] = "\n".join(f'<tr><td><code>{E(k.replace(" ", "␣"))}</code></td><td>' + ", ".join(f'<code>{E(w.replace(" ", "␣"))}</code> {s:.2f}' for w, s in v) + "</td></tr>" for k, v in D["gpt2"]["nn"].items())
G["analogy"] = "\n".join(f'<tr><td><code>{E(a.strip())}</code> − <code>{E(b.strip())}</code> + <code>{E(c.strip())}</code></td><td>' + ", ".join(f'<code>{E(w.replace(" ", "␣"))}</code>' for w in r) + "</td></tr>" for a, b, c, r in D["gpt2"]["analogy"])
rows = ['<table>', '<tr><th>n</th><th>table cells V<sup>n+1</sup></th><th>cells seen in training</th><th>count n-gram: train / held-out</th><th>neural LM: train / held-out</th><th>neural LM parameters</th></tr>']
for r in S:
    ng = f'{r["ng_train"]:.2f} / {r["ng_test"]:.2f}' if "ng_test" in r else "&mdash;"
    nl = f'{r["mlp_train"]:.2f} / {r["mlp_test"]:.2f}' if "mlp_test" in r else "&mdash;"
    cells = f'{65 ** (r["n"] + 1):,}'; seen = f'{r["ng_entries"]:,}' if "ng_entries" in r else "&mdash;"
    rows.append(f'<tr><td>{r["n"]}</td><td>{cells}</td><td>{seen}</td><td>{ng}</td><td>{nl}</td><td>{r["mlp_params"]:,}</td></tr>'.replace("<td>None</td>", "<td>&mdash;</td>") if "mlp_params" in r else f'<tr><td>{r["n"]}</td><td>{cells}</td><td>{seen}</td><td>{ng}</td><td>{nl}</td><td>&mdash;</td></tr>')
rows.append("</table>")
G["sweep"] = "\n".join(rows)

# ---- patch the lesson
p = os.path.join(HERE, "..", "..", "docs", "03_embeddings_bigram.html"); h = open(p).read()
h = re.sub(r'\n?<figure class="fig" id="fig\d+">.*?</figure>\n', "", h, flags=re.S)
for key, fg in F.items():
    m = f"<!--fig:{key}-->"; assert h.count(m) == 1, key
    h = h.replace(m, m + "\n" + fg)
for k, v in G.items():
    pat = re.compile(rf"<!--gen:{k}-->.*?<!--/gen:{k}-->", re.S); assert len(pat.findall(h)) == 1, k
    h = pat.sub(lambda m: f"<!--gen:{k}-->{v}<!--/gen:{k}-->", h)
open(p, "w").write(h); print("figures:", h.count("<figure"))
