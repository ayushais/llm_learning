"""Extract the real data used by the module 4 figures (run once; writes data.json)."""
import os, sys, json, math, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from attn_core import *
torch.manual_seed(0); np.random.seed(0)
D = {}

# ---- 1. worked example: 4 tokens, d_k = 2 (small hand-picked numbers; the arithmetic is real) ----
q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
k = torch.tensor([[1., 0.], [0., 2.], [1., 1.], [-1., 1.]])
v = torch.tensor([[1., 0.], [0., 1.], [2., 2.], [0., -2.]])
raw = q @ k.T; sc = raw / math.sqrt(2)
out, w = attention(q, k, v)
D["toy"] = {"raw": raw.tolist(), "scaled": sc.tolist(), "w": w.tolist(), "out": out.tolist(), "v": v.tolist(),
            "w_unmasked": F.softmax(sc, -1).tolist()}

# ---- 2. why divide by sqrt(d_k): q, k with unit-variance entries ----
g = torch.Generator().manual_seed(0); rows = []
for dk in (4, 16, 64, 256, 1024):
    qq = torch.randn(4000, dk, generator=g); kk = torch.randn(4000, 32, dk, generator=g)   # 4000 queries x 32 keys
    s = torch.einsum("qd,qkd->qk", qq, kk)
    rows.append({"dk": dk, "std_raw": float(s.std()), "std_scaled": float((s / math.sqrt(dk)).std()),
                 "maxw_raw": float(F.softmax(s, -1).max(-1).values.mean()),
                 "maxw_scaled": float(F.softmax(s / math.sqrt(dk), -1).max(-1).values.mean())})
D["scale"] = rows

# ---- 3. GPT-2 small: from-scratch attention vs Hugging Face, head catalogue, attention sink ----
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
gm = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
text = open(os.path.join(HERE, "..", "..", "data", "tinyshakespeare.txt")).read()
enc = tok.encode(text[:200000]); T = 128
chunks = torch.tensor([enc[i * T:(i + 1) * T] for i in range(12)])
with torch.no_grad():
    o = gm(chunks, output_attentions=True, output_hidden_states=True)
A = torch.stack(o.attentions)                                   # (12 layers, 12 chunks, 12 heads, T, T)

# 3a. our attention reproduces layer 5, head-by-head weights and the module output
l = 5; blk = gm.transformer.h[l]
with torch.no_grad():
    x = blk.ln_1(o.hidden_states[l])                             # input to attention of layer l
    qkv = x @ blk.attn.c_attn.weight + blk.attn.c_attn.bias
    q_, k_, v_ = qkv.split(768, -1)
    sp = lambda z: z.view(12, T, 12, 64).transpose(1, 2)
    y, w_ = attention(sp(q_), sp(k_), sp(v_))
    mine = (y.transpose(1, 2).reshape(12, T, 768) @ blk.attn.c_proj.weight + blk.attn.c_proj.bias)
    h1 = o.hidden_states[l] + mine                                # residual add, then the block's own MLP half
    h2 = h1 + blk.mlp(blk.ln_2(h1)); ref = o.hidden_states[l + 1]
D["gpt2_check"] = {"layer": l, "max_abs_weight_diff": float((w_ - A[l]).abs().max()), "max_abs_block_diff": float((h2 - ref).abs().max()),
                   "n_heads": 12, "d_head": 64}

# 3b. head statistics on 12 x 128 tokens of Shakespeare
offs = torch.zeros(12, 12)                                       # mean attention on the previous token
first = torch.zeros(12, 12)                                      # mean attention on token 0 (queries t >= 1)
ent = torch.zeros(12, 12)
idx = torch.arange(T)
for l in range(12):
    prev = A[l][:, :, idx[1:], idx[1:] - 1]; offs[l] = prev.mean((0, 2))
    first[l] = A[l][:, :, 1:, 0].mean((0, 2))
    ent[l] = (-(A[l] * torch.log(A[l].clamp_min(1e-9))).sum(-1)).mean((0, 2))
# induction score: random tokens repeated twice; mass from the 2nd copy at position t on the token after its previous occurrence
gr = torch.Generator().manual_seed(1); L0 = 50
rnd = torch.randint(1000, 40000, (8, L0), generator=gr); rep = torch.cat([rnd, rnd], 1)
with torch.no_grad(): Ar = torch.stack(gm(rep, output_attentions=True).attentions)
ind = torch.stack([Ar[l][:, :, torch.arange(L0, 2 * L0 - 1), torch.arange(1, L0)].mean((0, 2)) for l in range(12)])
pl, ph = divmod(int(offs.argmax()), 12); il, ih = divmod(int(ind.argmax()), 12); sl, sh = divmod(int(first[1:].argmax()), 12); sl += 1
# a diffuse head: highest mean entropy relative to its position (use layer 0 .. any)
dl, dh = divmod(int(ent.argmax()), 12)
D["heads"] = {"prev": {"l": pl, "h": ph, "score": float(offs[pl, ph])}, "ind": {"l": il, "h": ih, "score": float(ind[il, ih])},
              "sink": {"l": sl, "h": sh, "score": float(first[sl, sh])}, "diffuse": {"l": dl, "h": dh, "entropy": float(ent[dl, dh])}}
D["sink_by_layer"] = {"mean": first.mean(1).tolist(), "frac_heads_over_half": (first > 0.5).float().mean(1).tolist()}
D["induction_layers"] = ind.max(1).values.tolist()
D["baseline_uniform_prev"] = float((1 / torch.arange(2, T + 1).float()).mean())    # what a uniform head would put on the previous token

# 3c. show the four heads on one sentence containing a repeat
sent = "The cat sat on the mat because the cat was tired. The cat sat"
ids_s = tok(sent, return_tensors="pt").input_ids
with torch.no_grad(): As = gm(ids_s, output_attentions=True).attentions
toks = [tok.decode([i]) for i in ids_s[0]]
D["sentence"] = {"tokens": toks, "heads": {n: As[D["heads"][n]["l"]][0, D["heads"][n]["h"]].numpy().round(4).tolist() for n in ("prev", "ind", "sink", "diffuse")}}
# induction head behaviour on the real repeat: where does the final " sat" (2nd) look?
D["gpt2_text_loss_note"] = "see notebook"

# ---- 4. tiny attention LMs trained on TinyShakespeare ----
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
ids = np.array([stoi[c] for c in text]); cut = int(0.9 * len(ids)); tr, te = ids[:cut], ids[cut:]
TT = 64; runs = {}
for name, h, L in (("1 layer, 1 head", 1, 1), ("1 layer, 4 heads", 4, 1), ("2 layers, 4 heads", 4, 2)):
    torch.manual_seed(0); m = AttnLM(V, TT, 64, h, L); t0 = time.time(); train_attn(m, tr, steps=4000)
    runs[name] = {"test": eval_attn(m, te), "train": eval_attn(m, tr[:len(te)]), "params": sum(p.numel() for p in m.parameters())}
    print(name, runs[name], f"{time.time()-t0:.0f}s", flush=True)
    if name == "1 layer, 4 heads": m1 = m
    if name == "2 layers, 4 heads": m2 = m
D["runs"] = runs
# attention mass by distance for the 1-layer, 4-head model, averaged over held-out windows
xs = torch.as_tensor(te[: 200 * TT], dtype=torch.long).view(200, TT)
with torch.no_grad(): _, wl = m1(xs, True)
W1 = wl[0]                                                         # (200, 4, T, T)
prof = torch.zeros(4, 16)
for d in range(16):
    prof[:, d] = W1[:, :, torch.arange(16, TT), torch.arange(16, TT) - d].mean((0, 2))     # queries at t >= 16 so every distance up to 15 exists
D["offset_profile"] = prof.numpy().round(4).tolist()
D["samples"] = {"1 layer, 4 heads": sample_attn(m1, chars, stoi, "\n", 240, seed=1), "2 layers, 4 heads": sample_attn(m2, chars, stoi, "\n", 240, seed=1)}
m3 = json.load(open(os.path.join(HERE, "..", "module03", "data.json")))
D["baselines"] = {"bigram": m3["bigram"]["gd_test"], "mlp8": [r for r in m3["sweep"] if r["n"] == 8][0]["mlp_test"],
                  "ngram4": [r for r in m3["sweep"] if r["n"] == 4][0]["ng_test"]}
json.dump(D, open(os.path.join(HERE, "data.json"), "w"))
print(json.dumps({k: D[k] for k in ("gpt2_check", "heads", "sink_by_layer", "baselines", "runs")}, indent=1)[:3000])
print(D["samples"])
