"""Build the SVG figures of docs/04_self_attention.html from real data (run data.py first).
Idempotent: figures are re-inserted after the <!--fig:name--> markers."""
import os, re, json, math
from html import escape as E
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))
W = 480
NUM = {"pipe": 1, "toy": 2, "scale": 3, "heads": 4, "sink": 5, "offs": 6, "loss": 7}
ARROW = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'
def t(x, y, s, cls="", anchor="start", size=12, extra=""): return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{cls}" {extra}>{E(s)}</text>'
def svg(h, body, label): return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(label)}">{body}</svg>'
def fig(key, s, cap): n = NUM[key]; return f'<figure class="fig" id="fig{n}">{s}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n'
def arrow(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#ah)"/>'
def axes(ox, oy, pw, ph): return f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" class="axis"/>'
def box(x, y, w, h, col="var(--acc)"): return f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="5" style="stroke:{col}"/>'
F = {}

# 1 pipeline (schematic)
b = ARROW + f'<rect class="chip" x="6" y="62" width="44" height="40" rx="5"/>' + t(28, 79, "x", "mono", "middle", 13) + t(28, 94, "T×d", "mut", "middle", 11)
for i, (w_, o_, c) in enumerate((("W_q", "Q", "var(--s1)"), ("W_k", "K", "var(--s2)"), ("W_v", "V", "var(--s3)"))):
    y = 14 + i * 56
    b += arrow(50, 82, 70, y + 17) + box(72, y, 48, 34, c) + t(96, y + 22, w_, "mono", "middle", 12)
    b += arrow(122, y + 17, 142, y + 17) + f'<rect class="chip" x="144" y="{y}" width="40" height="34" rx="5"/>' + t(164, y + 22, o_, "mono", "middle", 13, 'font-weight="700"')
b += arrow(186, 31, 204, 52) + arrow(186, 87, 204, 66)
b += box(206, 24, 150, 62) + t(281, 44, "scores = QKᵀ / √d_k", "mono", "middle", 11) + t(281, 60, "causal mask", "", "middle", 11.5) + t(281, 76, "softmax per row → A", "", "middle", 11.5)
b += arrow(281, 88, 281, 114)
b += box(206, 116, 150, 34) + t(281, 138, "output = A · V", "mono", "middle", 12) + arrow(186, 143, 204, 138)
b += arrow(358, 133, 374, 133) + box(376, 116, 50, 34) + t(401, 138, "W_o", "mono", "middle", 12)
b += arrow(428, 133, 448, 133) + t(466, 138, "out", "mono", "middle", 12) + t(8, 184, "one head; multi-head runs h of these in parallel, each with its own W_q, W_k, W_v", "mut", "start", 11)
F["pipe"] = fig("pipe", svg(194, b, "Data flow of one self-attention head"),
    "The whole mechanism on one page. Three learned projections turn each token's vector into a query Q, a key K and a value V; query-key dot products give a T×T score table; a causal mask and a row-wise softmax turn it into weights A; the output is the weighted average of the values, followed by the output projection W_o. Schematic, not measured data.")

# 2 worked example: three 4x4 grids
T4 = D["toy"]; cs = 34; ox = [34, 188, 342]; oy = 44
b = ""
titles = ["1. raw scores q·k", "2. ÷√2, mask future", "3. softmax → weights"]
for p in range(3):
    b += t(ox[p] + 2 * cs, 14, titles[p], "", "middle", 11.5, 'font-weight="600"')
    for j in range(4): b += t(ox[p] + j * cs + cs / 2, oy - 6, f"k{j+1}", "mut", "middle", 11)
    for i in range(4):
        b += t(ox[p] - 4, oy + i * cs + cs / 2 + 4, f"q{i+1}", "mut", "end", 11)
        for j in range(4):
            x, y = ox[p] + j * cs, oy + i * cs
            if p == 0: v = T4["raw"][i][j]; txt = f"{v:g}"; fill = 'class="chip"'
            elif p == 1:
                if j > i: txt = "−∞"; fill = 'class="mask"'
                else: txt = f"{T4['scaled'][i][j]:.2f}"; fill = 'class="chip"'
            else:
                wv = T4["w"][i][j]; txt = f"{wv:.2f}"
                fill = f'class="cell" opacity="{0.15 + 0.75 * wv:.2f}"' if j <= i else 'class="mask"'
            b += f'<rect {fill} x="{x}" y="{y}" width="{cs-2}" height="{cs-2}" rx="3"/>'
            b += t(x + cs / 2 - 1, y + cs / 2 + 3, txt, "", "middle", 11, 'style="font-weight:600"' if p == 2 and j <= i and T4["w"][i][j] > 0.55 else "")
o = T4["out"]; b += t(8, oy + 4 * cs + 22, "output rows = weights · V:", "mono", "start", 11) + t(8, oy + 4 * cs + 38, "   ".join(f"o{i+1}=({o[i][0]:.2f}, {o[i][1]:.2f})" for i in range(4)), "mono", "start", 11)
F["toy"] = fig("toy", svg(oy + 4 * cs + 46, b, "Raw scores, masked scaled scores and softmax weights for four tokens"),
    "A worked example with four tokens and d<sub>k</sub> = 2 (small hand-picked vectors, so the arithmetic is easy to follow; the numbers are computed by the code of this module). <b>Left:</b> every query dotted with every key. <b>Middle:</b> divide by √2 and replace the entries above the diagonal by −∞ so a token cannot see later tokens. <b>Right:</b> softmax within each row; the first row is forced to put all its weight on itself, and each later row spreads over the tokens up to its own position. Illustrative values.")

# 3 scale factor
S = D["scale"]; xs = [r["dk"] for r in S]
def panel(ox, oy, pw, ph, ymax, key_raw, key_sc, title, fmt):
    X = lambda i: ox + (i + 0.5) / len(S) * pw; Y = lambda v: oy + ph - v / ymax * ph
    s = axes(ox, oy, pw, ph) + t(ox, oy - 8, title, "mut", "start", 11)
    for g in (0, ymax / 2, ymax): s += f'<line x1="{ox}" y1="{Y(g):.1f}" x2="{ox+pw}" y2="{Y(g):.1f}" class="grid"/>' + t(ox - 5, Y(g) + 4, fmt(g), "mut", "end", 11)
    for key, col in ((key_raw, "var(--s2)"), (key_sc, "var(--s1)")):
        s += f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{" ".join(f"{X(i):.1f},{Y(r[key]):.1f}" for i, r in enumerate(S))}"/>'
        for i, r in enumerate(S): s += f'<circle cx="{X(i):.1f}" cy="{Y(r[key]):.1f}" r="3.5" fill="{col}" stroke="var(--bg)" stroke-width="1.5"><title>d_k={r["dk"]}: {r[key]:.3f}</title></circle>'
    for i, r in enumerate(S): s += t(X(i), oy + ph + 15, str(r["dk"]), "mut", "middle", 11)
    s += t(ox + pw / 2, oy + ph + 30, "d_k", "mut", "middle", 11)
    return s, X, Y
a, X1, Y1 = panel(36, 34, 190, 130, 32, "std_raw", "std_scaled", "std of the score q·k", lambda g: f"{g:g}")
a += t(X1(2) - 6, Y1(S[2]["std_raw"]) - 10, "unscaled", "", "end", 11.5, 'style="fill:var(--s2)"') + t(X1(3), Y1(1) - 8, "÷√d_k", "", "middle", 11.5, 'style="fill:var(--s1)"')
c, X2, Y2 = panel(270, 34, 190, 130, 1.0, "maxw_raw", "maxw_scaled", "largest softmax weight (mean, 32 keys)", lambda g: f"{g:.1f}")
c += t(X2(1) - 8, Y2(S[1]["maxw_raw"]) - 8, "unscaled", "", "end", 11.5, 'style="fill:var(--s2)"') + t(X2(3), Y2(0.16) - 8, "÷√d_k", "", "middle", 11.5, 'style="fill:var(--s1)"')
F["scale"] = fig("scale", svg(212, a + c, "Score spread and softmax sharpness versus key dimension, with and without the 1/sqrt(d_k) scaling"),
    "Why the scores are divided by √d<sub>k</sub>. Queries and keys with independent unit-variance entries (4,000 queries × 32 keys per setting) have dot products whose standard deviation grows like √d<sub>k</sub> (left, orange). A softmax over such large scores is nearly one-hot (right, orange: for d<sub>k</sub>=1024 the biggest of 32 weights is already about 0.95), where gradients vanish. Dividing by √d<sub>k</sub> keeps the spread at 1 and the softmax equally soft for every d<sub>k</sub> (blue). Simulation with random Gaussian vectors, not trained weights.")

# 4 four GPT-2 heads on one sentence
Sn = D["sentence"]; toks = Sn["tokens"]; n = len(toks); cs = 11; px = [46, 276]; py = [30, 232]
names = [("prev", "a previous-token head"), ("ind", "an induction head"), ("sink", "a first-token (sink) head"), ("diffuse", "a diffuse head")]
b = ""
for k, (key, nm) in enumerate(names):
    ox, oy = px[k % 2], py[k // 2]; hd = D["heads"][key]
    b += t(ox - 40, oy - 14, f"layer {hd['l']}, head {hd['h']}: {nm}", "", "start", 11, 'font-weight="600"')
    for i, tk in enumerate(toks): b += t(ox - 3, oy + i * cs + 9, tk.strip() or "·", "mut", "end", 9.5 if False else 11) if False else ""
    for i in range(n):
        if i % 2 == 0 or i == n - 1: b += t(ox - 3, oy + i * cs + 9, toks[i].strip(), "mut", "end", 11)
    b += f'<rect x="{ox}" y="{oy}" width="{n*cs}" height="{n*cs}" class="mask"/>'
    for i in range(n):
        for j in range(i + 1):
            wv = Sn["heads"][key][i][j]
            if wv < 0.01: continue
            b += f'<rect class="cell" x="{ox+j*cs}" y="{oy+i*cs}" width="{cs-0.5}" height="{cs-0.5}" opacity="{min(1, wv**0.7):.2f}"><title>"{E(toks[i].strip())}" attends to "{E(toks[j].strip())}": {wv:.2f}</title></rect>'
F["heads"] = fig("heads", svg(py[1] + n * cs + 12, b, "Attention weights of four GPT-2 heads on one sentence"),
    f"Four of GPT-2 small's 144 heads on the sentence “{E(Sn['tokens'][0])}{E(''.join(Sn['tokens'][1:]))}”. Rows are the query tokens (every second one is labelled), columns the same tokens as keys, darker = more weight; the empty upper triangle is the causal mask. The heads were chosen by score: <b>L{D['heads']['prev']['l']}H{D['heads']['prev']['h']}</b> puts {D['heads']['prev']['score']*100:.0f}% of its mass on the previous token (averaged over 1,536 tokens of Shakespeare); <b>L{D['heads']['ind']['l']}H{D['heads']['ind']['h']}</b> is the best induction head (see below); <b>L{D['heads']['sink']['l']}H{D['heads']['sink']['h']}</b> sends {D['heads']['sink']['score']*100:.0f}% of its mass to the very first token; <b>L{D['heads']['diffuse']['l']}H{D['heads']['diffuse']['h']}</b> has the highest average entropy of all heads. Hover cells for values. Real GPT-2 weights.")

# 5 attention sink by layer
sk = D["sink_by_layer"]; ox, oy, pw, ph = 46, 26, 400, 150
X = lambda l: ox + (l + 0.5) / 12 * pw; Y = lambda v: oy + ph - v * ph
b = axes(ox, oy, pw, ph) + t(ox, oy - 10, "share of attention given to the first token", "mut", "start", 11)
for g in (0, 0.25, 0.5, 0.75, 1.0): b += f'<line x1="{ox}" y1="{Y(g):.1f}" x2="{ox+pw}" y2="{Y(g):.1f}" class="grid"/>' + t(ox - 6, Y(g) + 4, f"{g:.2f}".rstrip("0").rstrip("."), "mut", "end", 11)
for key, col in (("mean", "var(--s1)"), ("frac_heads_over_half", "var(--s2)")):
    b += f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{" ".join(f"{X(l):.1f},{Y(v):.1f}" for l, v in enumerate(sk[key]))}"/>'
    for l, v in enumerate(sk[key]): b += f'<circle cx="{X(l):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}" stroke="var(--bg)" stroke-width="1.5"><title>layer {l}: {v:.2f}</title></circle>'
b += t(X(5), Y(0.3), "mean over all heads", "", "start", 11.5, 'style="fill:var(--s1)"') + t(X(0) + 4, Y(0.78), "fraction of heads with >50% on token 0", "", "start", 11.5, 'style="fill:var(--s2)"')
for l in range(12): b += t(X(l), oy + ph + 15, str(l), "mut", "middle", 11)
b += t(ox + pw / 2, oy + ph + 31, "layer", "mut", "middle", 11)
F["sink"] = fig("sink", svg(oy + ph + 38, b, "Attention paid to the first token in each layer of GPT-2 small"),
    "The first token soaks up attention. For each layer of GPT-2 small: the average share of attention that queries at positions ≥ 1 give to position 0 (blue), and the fraction of the layer's 12 heads that give it more than half (orange). Measured on 12 windows of 128 tokens of Shakespeare. The first token is rarely the most informative one; it acts as a parking place when a head has nothing useful to attend to (“attention sink”, Xiao et al. 2023). Real GPT-2 weights.")

# 6 learned offset profiles of the tiny model
prof = D["offset_profile"]; b = ""; pw, ph = 190, 70; pos = [(34, 28), (268, 28), (34, 148), (268, 148)]
for h, (ox, oy) in enumerate(pos):
    b += axes(ox, oy, pw, ph) + t(ox, oy - 8, f"head {h+1}", "", "start", 11.5, 'font-weight="600"')
    bw = pw / 16
    for d, v in enumerate(prof[h]): b += f'<rect class="bar" x="{ox+d*bw+1:.1f}" y="{oy+ph-v/0.7*ph:.1f}" width="{bw-2:.1f}" height="{v/0.7*ph:.1f}"><title>head {h+1}, {d} characters back: {v:.3f}</title></rect>'
    for d in (0, 5, 10, 15): b += t(ox + d * bw + bw / 2, oy + ph + 14, str(d), "mut", "middle", 11)
    b += t(ox - 5, oy + 4, "0.7", "mut", "end", 11) + t(ox - 5, oy + ph + 4, "0", "mut", "end", 11)
b += t(250, 140, "", "mut") + t(ox + 0, 0, "")
b = b.replace(t(250, 140, "", "mut"), "")
b += t(240, 262, "distance back from the query (0 = the character itself)", "mut", "middle", 11)
F["offs"] = fig("offs", svg(270, b, "Average attention by distance for the four heads of a tiny trained character model"),
    "What four heads of a 29,121-parameter character-level model learned (1 layer, 4 heads, d=64, trained 4,000 steps on TinyShakespeare). Each panel is the average attention weight a head puts on the character <i>d</i> positions back, over 200 held-out windows and queries at position ≥ 16. Head 2 is mostly a “previous character” head; the others spread over the last three or four characters. Nothing told the model to do this. Real run, one seed.")

# 7 loss comparison
R = D["runs"]; Bs = D["baselines"]
rows = [("bigram (module 3)", Bs["bigram"], "bar2"), ("attention, 1 layer, 1 head", R["1 layer, 1 head"]["test"], "bar"), ("attention, 1 layer, 4 heads", R["1 layer, 4 heads"]["test"], "bar"),
        ("attention, 2 layers, 4 heads", R["2 layers, 4 heads"]["test"], "bar"), ("neural LM, 8 chars (module 3)", Bs["mlp8"], "bar2"), ("count 4-gram (module 3)", Bs["ngram4"], "bar2")]
ox, oy, bh = 190, 26, 28; scale = 270 / 2.6
b = t(8, 14, "held-out cross-entropy, nats per character (lower is better)", "mut", "start", 11)
for i, (nm, v, cl) in enumerate(rows):
    y = oy + i * (bh + 6)
    b += t(ox - 8, y + 18, nm, "", "end", 11.5) + f'<rect class="{cl}" x="{ox}" y="{y}" width="{v*scale:.1f}" height="{bh}"><title>{nm}: {v:.3f}</title></rect>' + t(ox + v * scale + 6, y + 18, f"{v:.2f}", "", "start", 11.5)
F["loss"] = fig("loss", svg(oy + len(rows) * (bh + 6) + 4, b, "Held-out loss of small attention models compared with the module 3 baselines"),
    "Tiny attention-only models (blue) against the module 3 baselines (orange), all scored on the same held-out Shakespeare text, predictions at positions ≥ 8 only. Four heads beat one; a second layer beats both and approaches the best baselines. The attention models have no MLP or normalisation yet (Module 5 adds them) and only 4,000 training steps, so this shows the direction of the effect, not what attention can do. Real runs, one seed each.")

# ---- generated text
G = {}
G["sample_attn"] = E(D["samples"]["2 layers, 4 heads"].lstrip("\n"))
G["runs"] = "<table>\n<tr><th>model</th><th>parameters</th><th>train loss</th><th>held-out loss</th></tr>\n" + "\n".join(
    f'<tr><td>{E(k)}</td><td>{v["params"]:,}</td><td>{v["train"]:.3f}</td><td>{v["test"]:.3f}</td></tr>' for k, v in R.items()) + "\n</table>"
Gc = D["gpt2_check"]
G["check"] = f'layer {Gc["layer"]}: largest difference in any attention weight {Gc["max_abs_weight_diff"]:.1e}; largest difference in the block output {Gc["max_abs_block_diff"]:.1e}'

p = os.path.join(HERE, "..", "..", "docs", "04_self_attention.html"); h = open(p).read()
h = re.sub(r'\n?<figure class="fig" id="fig\d+">.*?</figure>\n', "", h, flags=re.S)
for key, fg in F.items():
    m = f"<!--fig:{key}-->"; assert h.count(m) == 1, key
    h = h.replace(m, m + "\n" + fg)
for k, v in G.items():
    pat = re.compile(rf"<!--gen:{k}-->.*?<!--/gen:{k}-->", re.S); assert len(pat.findall(h)) == 1, k
    h = pat.sub(lambda m: f"<!--gen:{k}-->{v}<!--/gen:{k}-->", h)
open(p, "w").write(h); print("figures:", h.count("<figure"))
