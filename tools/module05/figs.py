"""Build the SVG figures of docs/05_transformer_block.html from real data (run data.py first).
Idempotent: figures are re-inserted after the <!--fig:name--> markers, generated text between <!--gen:k--> markers."""
import os, re, json, math
from html import escape as E
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))
W = 480
NUM = {"block": 1, "grad": 2, "train": 3, "act": 4, "stream": 5, "lens": 6, "abl": 7}
ARROW = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--acc)"/></marker></defs>'
def t(x, y, s, cls="", anchor="start", size=12, extra=""): return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" class="{cls}" {extra}>{E(s)}</text>'
def svg(h, body, label): return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(label)}">{body}</svg>'
def fig(key, s, cap): n = NUM[key]; return f'<figure class="fig" id="fig{n}">{s}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n'
def arrow(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#ah)"/>'
def line(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln"/>'
def axes(ox, oy, pw, ph): return f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" class="axis"/>'
def box(x, y, w, h, col="var(--acc)"): return f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="5" style="stroke:{col}"/>'
def plus(cx, cy): return f'<circle cx="{cx}" cy="{cy}" r="8" class="chip" style="stroke:var(--acc);stroke-width:1.5"/>' + t(cx, cy + 4.5, "+", "", "middle", 13, 'font-weight="700"')
def hgrid(ox, pw, Y, vals, fmt):
    return "".join(f'<line x1="{ox}" y1="{Y(v):.1f}" x2="{ox+pw}" y2="{Y(v):.1f}" class="grid"/>' + t(ox - 5, Y(v) + 4, fmt(v), "mut", "end", 11) for v in vals)
def series(pts, col, tip, r=3):
    s = f'<polyline fill="none" stroke="{col}" stroke-width="2.2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)}"/>'
    return s + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" stroke="var(--bg)" stroke-width="1.2"><title>{E(tip(lab))}</title></circle>' for x, y, lab in pts if r)
COL = {"pre-LN": "var(--s1)", "post-LN": "var(--s2)", "no residual": "var(--s3)", "post-LN + warmup": "var(--s2)"}
F, G = {}, {}

# 1 post-norm vs pre-norm (schematic)
NC, SC = "var(--s2)", "var(--s1)"
def panel_post(ox):
    sx, bx = ox + 40, ox + 140
    s = t(sx, 22, "x", "mono", "middle", 13) + line(sx, 28, sx, 92)
    s += line(sx, 46, bx, 46) + arrow(bx, 46, bx, 56) + box(bx - 38, 58, 76, 26, SC) + t(bx, 75, "Attn", "", "middle", 12)
    s += line(bx, 84, bx, 100) + arrow(bx, 100, sx + 10, 100) + plus(sx, 100)
    s += arrow(sx, 108, sx, 118) + box(sx - 26, 120, 52, 24, NC) + t(sx, 136, "Norm", "", "middle", 11.5)
    s += line(sx, 144, sx, 206) + line(sx, 160, bx, 160) + arrow(bx, 160, bx, 170) + box(bx - 38, 172, 76, 26, SC) + t(bx, 189, "FFN", "", "middle", 12)
    s += line(bx, 198, bx, 214) + arrow(bx, 214, sx + 10, 214) + plus(sx, 214)
    s += arrow(sx, 222, sx, 232) + box(sx - 26, 234, 52, 24, NC) + t(sx, 250, "Norm", "", "middle", 11.5) + arrow(sx, 258, sx, 272)
    s += t(sx, 286, "out", "mono", "middle", 12)
    s += t(ox + 100, 308, "post-norm (Vaswani 2017)", "", "middle", 12, 'font-weight="600"') + t(ox + 100, 324, "norm sits on the main path", "mut", "middle", 11)
    return s
def panel_pre(ox):
    sx, bx = ox + 40, ox + 140
    s = t(sx, 22, "x", "mono", "middle", 13) + line(sx, 28, sx, 106)
    s += line(sx, 40, bx, 40) + arrow(bx, 40, bx, 48) + box(bx - 30, 50, 60, 22, NC) + t(bx, 65, "Norm", "", "middle", 11.5)
    s += arrow(bx, 72, bx, 80) + box(bx - 38, 82, 76, 26, SC) + t(bx, 99, "Attn", "", "middle", 12)
    s += line(bx, 108, bx, 120) + arrow(bx, 120, sx + 10, 120) + plus(sx, 120)
    s += line(sx, 128, sx, 220) + line(sx, 142, bx, 142) + arrow(bx, 142, bx, 150) + box(bx - 30, 152, 60, 22, NC) + t(bx, 167, "Norm", "", "middle", 11.5)
    s += arrow(bx, 174, bx, 182) + box(bx - 38, 184, 76, 26, SC) + t(bx, 201, "FFN", "", "middle", 12)
    s += line(bx, 210, bx, 222) + arrow(bx, 222, sx + 10, 222) + plus(sx, 222)
    s += arrow(sx, 230, sx, 272) + t(sx, 286, "out", "mono", "middle", 12)
    s += t(sx + 8, 250, "identity path,", "mut", "start", 11) + t(sx + 8, 263, "never normalised", "mut", "start", 11)
    s += t(ox + 100, 308, "pre-norm (GPT-2, Llama)", "", "middle", 12, 'font-weight="600"') + t(ox + 100, 324, "norm only on the branch", "mut", "middle", 11)
    return s
b = ARROW + panel_post(14) + f'<line x1="238" y1="14" x2="238" y2="296" class="grid"/>' + panel_pre(262)
F["block"] = fig("block", svg(334, b, "Post-norm and pre-norm transformer blocks"),
    "One block, two placements of the norm. <b>Left:</b> the original transformer normalises after each residual add, so the main path from input to output passes through 2N norms in an N-block model. <b>Right:</b> pre-norm (GPT-2, Llama) normalises only the input of each sublayer; the main path (the residual stream) is a pure sum, and a final norm is applied once after the last block. Schematic, not measured data.")

# 2 at init: gradient per block, similarity between positions
L = 12
def lineplot(ox, oy, pw, ph, data, ymin, ymax, logy, title, fmt, ticks, tip):
    X = lambda i: ox + (i + 0.5) / L * pw
    Y = (lambda v: oy + ph - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * ph) if logy else (lambda v: oy + ph - (v - ymin) / (ymax - ymin) * ph)
    s = axes(ox, oy, pw, ph) + t(ox, oy - 8, title, "mut", "start", 11) + hgrid(ox, pw, Y, ticks, fmt)
    for n, vals in data.items(): s += series([(X(i), Y(v), (n, i, v)) for i, v in enumerate(vals)], COL[n], lambda lab: f"{lab[0]}, block {lab[1]+1}: {lab[2]:.3f}")
    for i in (0, 3, 7, 11): s += t(X(i), oy + ph + 15, str(i + 1), "mut", "middle", 11)
    s += t(ox + pw / 2, oy + ph + 30, "block", "mut", "middle", 11)
    return s, X, Y
g = D["grad_init"]; c = D["cos_init"]
a, X1, Y1 = lineplot(40, 30, 180, 150, g, 0.1, 10, True, "gradient norm of each block", lambda v: f"{v:g}", (0.1, 1, 10), None)
a += t(X1(5), Y1(g["no residual"][5]) - 9, "no residual", "", "middle", 11.5, 'style="fill:var(--s3)"')
a += t(X1(6), Y1(g["post-LN"][6]) - 9, "post-LN", "", "middle", 11.5, 'style="fill:var(--s2)"') + t(X1(6), Y1(g["pre-LN"][6]) + 17, "pre-LN", "", "middle", 11.5, 'style="fill:var(--s1)"')
b_, X2, Y2 = lineplot(282, 30, 140, 150, c, 0, 1, False, "cos similarity between positions", lambda v: f"{v:g}", (0, 0.5, 1), None)
b_ += t(X2(2), Y2(0.72), "no residual: collapse", "", "start", 11.5, 'style="fill:var(--s3)"')
b_ += t(X2(11) + 8, Y2(c["post-LN"][11]) + 2, "post-LN", "", "start", 11.5, 'style="fill:var(--s2)"') + t(X2(11) + 8, Y2(c["pre-LN"][11]) + 12, "pre-LN", "", "start", 11.5, 'style="fill:var(--s1)"')
F["grad"] = fig("grad", svg(222, a + b_, "Gradient norm per block and similarity between positions at initialisation, 12-block models"),
    "Twelve-block character models at initialisation (d=64, LayerNorm, GELU MLP, one batch of 32×32 characters). <b>Left:</b> gradient norm of each block's parameters (log scale). Pre-norm is flat. Post-norm gradients are about twice as large but also roughly flat: the growth toward the output that Xiong et al. derive is not visible in a model this small, which fits Liu et al.’s point that gradient size alone does not explain why post-norm fails in Figure 3. Without residuals the gradients are larger still. <b>Right:</b> mean cosine similarity between the vectors of different positions after each block. Without residuals they become identical after about six blocks (rank collapse), so the output cannot depend on the context. Real measurements, one seed.")

# 3 training at lr 1e-2, and the lr sweep
dc = D["deep_curves"]; U = D["unigram"]
ox, oy, pw, ph = 40, 30, 210, 150; n = len(dc["pre-LN"]); X = lambda i: ox + (i + 1) / n * pw; Y = lambda v: oy + ph - (v - 1.8) / (4.0 - 1.8) * ph
b = axes(ox, oy, pw, ph) + t(ox, oy - 8, "train loss, learning rate 1e-2", "mut", "start", 11) + hgrid(ox, pw, Y, (2, 3, 4), lambda v: f"{v:g}")
b += f'<line x1="{ox}" y1="{Y(U):.1f}" x2="{ox+pw}" y2="{Y(U):.1f}" class="axis" stroke-dasharray="3 3"/>'
for name, vals in dc.items():
    pts = [(X(i), Y(min(v, 4.0)), (name, (i + 1) * 20, v)) for i, v in enumerate(vals) if v == v]
    b += series(pts, COL[name], lambda lab: f"{lab[0]}, step {lab[1]}: {lab[2]:.3f}", r=0).replace('stroke-width="2.2"', 'stroke-width="2.2" stroke-dasharray="5 3"' if name.endswith("warmup") else 'stroke-width="2.2"')
b += t(ox + pw - 2, Y(U) - 6, "unigram = post-LN = no residual", "mut", "end", 11)
b += t(X(n // 2), Y(dc["pre-LN"][n // 2]) + 18, f"pre-LN, ends at {dc['pre-LN'][-1]:.2f}", "", "middle", 11.5, 'style="fill:var(--s1)"') + t(X(5) + 4, Y(2.62), "post-LN + warmup", "", "start", 11, 'style="fill:var(--s2)"')
for s_ in (0, 200, 400): b += t(ox + s_ / 400 * pw, oy + ph + 15, str(s_), "mut", "middle", 11)
b += t(ox + pw / 2, oy + ph + 30, "step", "mut", "middle", 11)
sw = D["lr_sweep"]; lrs = sw["lrs"]; ox2, pw2 = 316, 150
X2 = lambda i: ox2 + (i + 0.5) / len(lrs) * pw2
b += axes(ox2, oy, pw2, ph) + t(ox2, oy - 8, "final train loss vs learning rate", "mut", "start", 11) + hgrid(ox2, pw2, Y, (2, 3, 4), lambda v: f"{v:g}")
b += f'<line x1="{ox2}" y1="{Y(U):.1f}" x2="{ox2+pw2}" y2="{Y(U):.1f}" class="axis" stroke-dasharray="3 3"/>'
for name in ("pre-LN", "post-LN"):
    b += series([(X2(i), Y(min(v, 4.0)), (name, lr, v)) for i, (lr, v) in enumerate(zip(lrs, sw[name]))], COL[name], lambda lab: f"{lab[0]}, lr {lab[1]:g}: {lab[2]:.3f}", r=3.5)
for i, lr in enumerate(lrs): b += t(X2(i), oy + ph + 15, f"{lr:g}", "mut", "middle", 11)
b += t(ox2 + pw2 / 2, oy + ph + 30, "learning rate", "mut", "middle", 11)
b += t(X2(2), Y(sw["post-LN"][2]) - 9, "post-LN", "", "middle", 11.5, 'style="fill:var(--s2)"') + t(X2(3), Y(sw["pre-LN"][3]) - 9, "pre-LN", "", "middle", 11.5, 'style="fill:var(--s1)"')
F["train"] = fig("train", svg(222, b, "Training curves of 12-block models at a high learning rate, and final loss against learning rate"),
    f"Training twelve-block models (d=64, Adam, batch 32×32 characters, 400 steps, cosine decay). <b>Left:</b> at learning rate 10<sup>−2</sup> without warmup, pre-norm (blue) learns; post-norm (orange, solid) and the model without residuals (aqua) stay at the unigram loss, {U:.2f} nats (dotted line); with a 100-step warmup (orange, dashed) post-norm starts to learn while the rate is small, then collapses back to the unigram loss as the rate reaches its peak. <b>Right:</b> final training loss at four learning rates. Post-norm trains only at the smallest, 10<sup>−3</sup>, where it is slightly ahead of pre-norm ({sw["post-LN"][0]:.2f} vs {sw["pre-LN"][0]:.2f}); from 3·10<sup>−3</sup> up it stays at the unigram loss. Pre-norm trains at all four. Real runs, one seed each.")

# 4 activation functions
A = D["act"]; z = A["z"]; ox, oy, pw, ph = 40, 20, 260, 150
X = lambda v: ox + (v + 4) / 7 * pw; Y = lambda v: oy + ph - (v + 1) / 4 * ph
b = axes(ox, oy, pw, ph) + hgrid(ox, pw, Y, (-1, 0, 1, 2, 3), lambda v: f"{v:g}")
for v in (-4, -2, 0, 2): b += t(X(v), oy + ph + 15, f"{v:g}", "mut", "middle", 11)
b += t(ox + pw / 2, oy + ph + 30, "z", "mut", "middle", 11)
for key, col, nm in (("relu", "var(--s1)", "ReLU"), ("gelu", "var(--s2)", "GELU"), ("silu", "var(--s3)", "Swish / SiLU")):
    b += f'<polyline fill="none" stroke="{col}" stroke-width="2.2" points="{" ".join(f"{X(a_):.1f},{Y(v):.1f}" for a_, v in zip(z, A[key]))}"><title>{nm}</title></polyline>'
i_m = min(range(len(z)), key=lambda i: A["silu"][i])
for k_, (nm, col) in enumerate((("ReLU", "s1"), ("GELU", "s2"), ("Swish / SiLU", "s3"))):
    b += f'<line x1="{X(-3.7):.1f}" y1="{Y(2.75 - 0.42*k_):.1f}" x2="{X(-3.2):.1f}" y2="{Y(2.75 - 0.42*k_):.1f}" stroke="var(--{col})" stroke-width="2.5"/>' + t(X(-3.1), Y(2.75 - 0.42*k_) + 4, nm, "", "start", 11.5, f'style="fill:var(--{col})"')
# right: the gated unit
gx = 330
b += t(gx, 30, "one SwiGLU hidden unit:", "", "start", 11.5, 'font-weight="600"')
b += t(gx, 52, "gate  a = x·w", "mono", "start", 11) + t(gx, 70, "value u = x·v", "mono", "start", 11)
b += t(gx, 92, "out = Swish(a) · u", "mono", "start", 11)
b += t(gx, 118, "plain MLP unit:", "", "start", 11.5, 'font-weight="600"') + t(gx, 138, "out = GELU(x·w)", "mono", "start", 11)
b += t(gx, 164, "the gate is a smooth", "mut", "start", 11) + t(gx, 178, "switch on u: off for", "mut", "start", 11) + t(gx, 192, "a ≪ 0, ≈ a·u for a ≫ 0", "mut", "start", 11)
F["act"] = fig("act", svg(214, b, "ReLU, GELU and Swish activation functions, and the form of a gated unit"),
    f"<b>Left:</b> the three activations, computed. GELU (z·Φ(z)) and Swish (z·σ(z), β=1) are smooth versions of ReLU with a small negative dip (Swish's minimum is {min(A['silu']):.3f} at z ≈ {z[i_m]:.1f}). <b>Right:</b> the difference that matters in SwiGLU is not the curve but the product: each hidden unit multiplies a gated projection by a second, linear projection of the same input.")

# 5 GPT-2 residual stream
S = D["stream"]; nl = 13
ox, oy, pw, ph = 46, 30, 180, 150; X = lambda l: ox + (l + 0.5) / nl * pw
Y = lambda v: oy + ph - (math.log10(v) - 0) / (4 - 0) * ph
b = axes(ox, oy, pw, ph) + t(ox, oy - 8, "norm of a token's stream vector", "mut", "start", 11) + hgrid(ox, pw, Y, (1, 10, 100, 1000, 10000), lambda v: f"{v:,.0f}")
b += series([(X(l), Y(v), (l, v)) for l, v in enumerate(S["norm_rest"])], "var(--s1)", lambda lab: f"after block {lab[0]}: {lab[1]:.1f}")
b += series([(X(l), Y(v), (l, v)) for l, v in enumerate(S["norm_first"])], "var(--s2)", lambda lab: f"token 0, after block {lab[0]}: {lab[1]:.0f}")
b += t(X(7), Y(S["norm_first"][7]) - 10, "token 0", "", "middle", 11.5, 'style="fill:var(--s2)"') + t(X(7), Y(S["norm_rest"][7]) + 18, "other tokens", "", "middle", 11.5, 'style="fill:var(--s1)"')
for l in (0, 4, 8, 12): b += t(X(l), oy + ph + 15, str(l), "mut", "middle", 11)
b += t(ox + pw / 2, oy + ph + 30, "after block (0 = embedding)", "mut", "middle", 11)
up = S["upd"]; ox2, pw2 = 296, 170; X2 = lambda l: ox2 + (l + 0.5) / 12 * pw2; Y2 = lambda v: oy + ph - min(v, 1.0) / 1.0 * ph
b += axes(ox2, oy, pw2, ph) + t(ox2, oy - 8, "|sublayer output| / |stream|", "mut", "start", 11) + hgrid(ox2, pw2, Y2, (0, 0.5, 1), lambda v: f"{v:g}")
for key, col in (("attn", "var(--s1)"), ("mlp", "var(--s2)")):
    b += series([(X2(l), Y2(u[key]), (l, u[key])) for l, u in enumerate(up)], col, lambda lab, k=key: f"{k}, block {lab[0]+1}: {lab[1]:.3f}" + (" (clipped)" if lab[1] > 1 else ""))
b += t(X2(5), Y2(0.45), "MLP", "", "middle", 11.5, 'style="fill:var(--s2)"') + t(X2(4), Y2(0.04) , "attention", "", "middle", 11.5, 'style="fill:var(--s1)"')
for l in (0, 5, 11): b += t(X2(l), oy + ph + 15, str(l + 1), "mut", "middle", 11)
b += t(ox2 + pw2 / 2, oy + ph + 30, "block", "mut", "middle", 11)
F["stream"] = fig("stream", svg(222, b, "Norm of GPT-2's residual stream by layer, and the relative size of each sublayer's update"),
    f"GPT-2 small's residual stream on 8 windows of 256 tokens of Shakespeare. <b>Left</b> (log scale): the norm of a token's vector after each block; blue is the average over positions 1–255, orange is position 0 (here the token “{E(S['first_token_text'])}”), which jumps to about {S['norm_first'][3]:,.0f} by block 3 and stays near {S['norm_first'][6]:,.0f} until the last block. <b>Right:</b> the size of what each sublayer adds, relative to the stream it is added to (positions ≥ 1). Values above 1 are clipped: block 1's attention ({up[0]['attn']:.1f}) writes into a stream that holds only the small embedding, and block 12's attention ({up[11]['attn']:.1f}) makes a large final edit. Real GPT-2 weights.")

# 6 logit lens
Ls = D["lens"]["layers"]; ox, oy, pw, ph = 46, 30, 180, 150; X = lambda l: ox + (l + 0.5) / 13 * pw
Y = lambda v: oy + ph - (v - 3) / (12 - 3) * ph
b = axes(ox, oy, pw, ph) + t(ox, oy - 8, "next-token loss (nats)", "mut", "start", 11) + hgrid(ox, pw, Y, (3, 6, 9, 12), lambda v: f"{v:g}")
pts = [(X(l), Y(min(r["loss"], 12)), (l, r["loss"])) for l, r in enumerate(Ls)]
b += series(pts, "var(--s1)", lambda lab: f"after block {lab[0]}: {lab[1]:.2f}" + (" (clipped)" if lab[1] > 12 else ""))
b += t(X(2), Y(11.2), f"embedding: {Ls[0]['loss']:.0f}, off the chart", "mut", "start", 11)
b += t(X(12) - 8, Y(Ls[12]["loss"]) + 4, f"final {Ls[12]['loss']:.2f}", "", "end", 11.5, 'style="fill:var(--s1)"')
for l in (0, 4, 8, 12): b += t(X(l), oy + ph + 15, str(l), "mut", "middle", 11)
b += t(ox + pw / 2, oy + ph + 30, "after block", "mut", "middle", 11)
ox2, pw2 = 296, 170; X2 = lambda l: ox2 + (l + 0.5) / 13 * pw2; Y2 = lambda v: oy + ph - v * ph
b += axes(ox2, oy, pw2, ph) + t(ox2, oy - 8, "top-1 same as the final model", "mut", "start", 11) + hgrid(ox2, pw2, Y2, (0, 0.5, 1), lambda v: f"{v:g}")
b += series([(X2(l), Y2(r["agree"]), (l, r["agree"])) for l, r in enumerate(Ls)], "var(--s2)", lambda lab: f"after block {lab[0]}: {lab[1]:.2f}")
for l in (0, 4, 8, 12): b += t(X2(l), oy + ph + 15, str(l), "mut", "middle", 11)
b += t(ox2 + pw2 / 2, oy + ph + 30, "after block", "mut", "middle", 11)
F["lens"] = fig("lens", svg(222, b, "Logit lens on GPT-2 small: loss and agreement with the final prediction when decoding each layer"),
    f"The logit lens on GPT-2 small: decode the residual stream after each block with the model's own final LayerNorm and unembedding, on the same 2,040 next-token predictions as Figure 5. <b>Left:</b> cross-entropy against the true next token (the embedding alone, {Ls[0]['loss']:.0f} nats, is off the chart; a uniform guess would be {D['lens']['uniform']:.1f}). <b>Right:</b> how often the top-1 token at that depth equals the final model's top-1. Real GPT-2 weights.")

# 7 ablation
R = D["runs"]; Bs = D["baselines"]
mean = lambda v: sum(v) / len(v)
rows = [("bigram (Module 3)", Bs["bigram"], None, "bar2")] + [(k, mean(v["test"]), v["test"], "bar") for k, v in R.items()] + [("count 4-gram (Module 3)", Bs["ngram4"], None, "bar2")]
ox, oy, bh = 214, 26, 26; vmax = 2.6; scale = 210 / vmax
b = t(8, 14, "held-out cross-entropy, nats per character (lower is better)", "mut", "start", 11)
for i, (nm, v, seeds, cl) in enumerate(rows):
    y = oy + i * (bh + 6)
    tip = f"{nm}: {v:.3f}" + (f" (seeds: {', '.join(f'{s:.3f}' for s in seeds)})" if seeds else "")
    b += t(ox - 8, y + 17, nm, "", "end", 11.5) + f'<rect class="{cl}" x="{ox}" y="{y}" width="{v*scale:.1f}" height="{bh}"><title>{E(tip)}</title></rect>' + t(ox + v * scale + 6, y + 17, f"{v:.2f}", "", "start", 11.5)
F["abl"] = fig("abl", svg(oy + len(rows) * (bh + 6) + 4, b, "Held-out loss as the ingredients of the block are added"),
    "Two-block character models (blue, mean of two seeds; hover for each seed) against the Module 3 baselines (orange), all on the same held-out Shakespeare text, predictions at positions ≥ 8. Each blue row adds one ingredient to the row above, except the last, which swaps LayerNorm → RMSNorm and GELU MLP → SwiGLU at matched parameter count. Real runs, 2,000 steps each.")

# ---- generated text
G["nores_loss"] = f"{dc['no residual'][-1]:.2f}"
sp = D["speed_ms"]
G["speed"] = f"{sp['ours_rms']:.0f} ms against {sp['ours_ln']:.0f} ms for our unfused LayerNorm, about half, as the formula suggests; but PyTorch's built-in <code>nn.LayerNorm</code>, a fused kernel, takes {sp['torch_ln']:.0f} ms and its <code>nn.RMSNorm</code> {sp['torch_rms']:.0f} ms on this CPU (PyTorch {E('2.14')}). On GPUs, fused RMSNorm kernels are the norm in LLM inference engines, and that is where the saving is realised"
sw_ = D["ln_swap"]
G["swap"] = f"{sw_['n_norms']} LayerNorms of a trained GPT-2 small by RMSNorm (same gain and bias, only the mean subtraction removed) raises its loss on our Shakespeare windows from {sw_['loss_ln']:.2f} to {sw_['loss_rms']:.2f} nats, and the top-1 prediction changes on {100*(1-sw_['top1_agree']):.0f}% of tokens"
gc = D["gpt2_check"]
G["check"] = f"the largest difference from the Hugging Face block output is {gc['max_abs_diff']:.1e}, on a stream whose entries reach {gc['stream_scale']:,.0f}"
G["growth"] = (f"The average norm of a token vector goes from {S['norm_rest'][1]:.0f} after block 1 to {S['norm_rest'][12]:.0f} after block 12 (the embedding itself is only {S['norm_rest'][0]:.1f}). "
               f"Yet from block 2 to block 11 every attention and MLP sublayer adds a vector whose norm is only {min(min(u['attn'], u['mlp']) for u in up[1:11]):.2f}–{max(max(u['attn'], u['mlp']) for u in up[1:11]):.2f} of the stream's (Figure 5, right); only the first block, which builds on the bare embedding, and the last attention layer make large edits.")
G["massive"] = (f"from block 3 on its norm is in the thousands (about {S['norm_first'][6]:,.0f} after block 6, against {S['norm_rest'][6]:.0f} for the others). Almost all of it sits in a few coordinates: after block 6, dimension {S['massive_dims'][0]} holds {S['massive_vals'][0]:,.0f} and dimension {S['massive_dims'][1]} holds {S['massive_vals'][1]:,.0f}, "
                f"while the median absolute entry of ordinary tokens is {S['other_typical_abs']:.2f}.")
G["meanrms"] = f"{max(S['mean_over_rms'][1:]):.2f}"
rr = {k: mean(v["test"]) for k, v in R.items()}; keys = list(R)
G["runs"] = "<table>\n<tr><th>model</th><th>parameters</th><th>train loss</th><th>held-out loss (seed 0 / 1)</th></tr>\n" + "\n".join(
    f'<tr><td>{E(k)}</td><td>{v["params"]:,}</td><td>{mean(v["train"]):.3f}</td><td>{v["test"][0]:.3f} / {v["test"][1]:.3f}</td></tr>' for k, v in R.items()) + "\n</table>"
G["sample"] = E(D["sample"].lstrip("\n"))
a0, a1, a2, a3 = (rr[k] for k in keys); spread = max(abs(v["test"][0] - v["test"][1]) for v in R.values())
G["abl_text"] = (f"Read it top to bottom. <b>The MLP is the big step:</b> {a0:.2f} → {a1:.2f} nats, with the parameter count going from {R[keys[0]]['params']:,} to {R[keys[1]]['params']:,}; the model now has somewhere to compute on what attention gathered, and it moves past the count-based 4-gram ({Bs['ngram4']:.2f}) that the attention-only model of Module 4 could not reach. "
    f"<b>Pre-LayerNorm does nothing for a 2-block model</b> ({a2:.2f} vs {a1:.2f}, while the two seeds of one configuration differ by up to {spread:.2f}): normalisation buys stability at depth and at high learning rates (Figures 2 and 3), not a better 2-block model. "
    f"<b>RMSNorm + SwiGLU</b> at the same parameter count gives the lowest loss, {a3:.2f}, {a2 - a3:.2f} below the GPT-2 block, and both of its seeds beat both GPT-2-block seeds. That is consistent with Shazeer's result but small, from two seeds in one tiny setting; treat it as a direction, not a measurement of the gain at scale.")

p = os.path.join(HERE, "..", "..", "docs", "05_transformer_block.html"); h = open(p).read()
h = re.sub(r'\n?<figure class="fig" id="fig\d+">.*?</figure>\n', "", h, flags=re.S)
for key, fg in F.items():
    m = f"<!--fig:{key}-->"; assert h.count(m) == 1, key
    h = h.replace(m, m + "\n" + fg)
for k, v in G.items():
    pat = re.compile(rf"<!--gen:{k}-->.*?<!--/gen:{k}-->", re.S); assert len(pat.findall(h)) == 1, k
    h = pat.sub(lambda m: f"<!--gen:{k}-->{v}<!--/gen:{k}-->", h)
open(p, "w").write(h); print("figures:", h.count("<figure"))
print({k: round(v, 3) for k, v in rr.items()})
