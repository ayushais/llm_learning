"""Extract the real data used by the module 3 figures (run once; writes data.json)."""
import os, sys, json, time, urllib.request, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lm_core import *
torch.manual_seed(0); np.random.seed(0)
D = {}

p = os.path.join(HERE, "..", "..", "data", "tinyshakespeare.txt")
if not os.path.exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", p)
text = open(p).read()
chars, stoi = make_vocab(text); V = len(chars)
ids = np.array([stoi[c] for c in text])
cut = int(len(ids) * 0.9); tr, te = ids[:cut], ids[cut:]
vcut = int(len(tr) * 0.9); tr_fit, va = tr[:vcut], tr[vcut:]       # validation slice only used to pick smoothing a
D["meta"] = {"V": V, "chars": chars, "n_train": len(tr), "n_test": len(te), "lnV": math.log(V)}

# ---- 1. bigram: gradient descent on a V x V table vs closed-form counts ----
X, Y = torch.tensor(tr[:-1]), torch.tensor(tr[1:])
Xt, Yt = torch.tensor(te[:-1]), torch.tensor(te[1:])
cnt = torch.zeros(V, V); cnt.index_put_((X, Y), torch.ones(len(X)), accumulate=True)
P_mle = cnt / cnt.sum(1, keepdim=True)
mle_train = float(-(torch.log(P_mle[X, Y])).mean())
uni = np.bincount(tr, minlength=V) / len(tr)
D["unigram_test"] = float(-np.log(uni[te]).mean())
W = torch.zeros(V, V, requires_grad=True)                          # logits table: row = previous char
opt = torch.optim.Adam([W], lr=0.05)
curve = []
for step in range(301):
    loss = F.cross_entropy(W[X], Y)
    if step in (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 250, 300): curve.append([step, float(loss)])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    gd_train = float(F.cross_entropy(W[X], Y)); gd_test = float(F.cross_entropy(W[Xt], Yt))
    Psm = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True)            # tiny smoothing so unseen test pairs are not -inf
    mle_test_sm = float(-(torch.log(Psm[Xt, Yt])).mean())
    maxdiff = float((F.softmax(W, 1) - P_mle).abs().max())
D["bigram"] = {"curve": curve, "mle_train": mle_train, "gd_train": gd_train, "gd_test": gd_test, "mle_test_smoothed": mle_test_sm,
               "max_abs_prob_diff": maxdiff, "zero_pairs": int((cnt == 0).sum()),
               "P": P_mle.numpy().round(4).tolist(), "freq": uni.round(5).tolist()}
class BigramLM(nn.Module):
    n = 1
    def __init__(s, W): super().__init__(); s.W = W
    def forward(s, x): return s.W[x[:, -1]]
D["bigram"]["sample"] = sample(BigramLM(W.detach()), chars, stoi, "\n", 240, seed=1)

# ---- 2. context-length sweep: count n-gram vs Bengio-style neural LM ----
sweep = []
for n in range(0, 9):
    row = {"n": n}
    if n <= 6:
        t0 = time.time(); ng = CountNgram(tr_fit, n, V)
        best = min((ng.nll(va, a), a) for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0))
        ng = CountNgram(tr, n, V)                                    # refit on all training text with the chosen a
        row.update(ng_alpha=best[1], ng_train=ng.nll(tr, best[1]), ng_test=ng.nll(te, best[1]), ng_contexts=int(len(ng.ctx_u)),
                   ng_entries=int(len(ng.key_u)), ng_possible=float(V) ** n * V)
        print("ngram", n, {k: row[k] for k in ("ng_alpha", "ng_train", "ng_test", "ng_contexts")}, f"{time.time()-t0:.0f}s", flush=True)
    if n >= 1:
        Xn, Yn = windows(tr, n); Xe, Ye = windows(te, n)
        m = NeuralLM(V, n); t0 = time.time(); train_lm(m, Xn, Yn, steps=6000)
        row.update(mlp_train=eval_loss(m, Xn, Yn), mlp_test=eval_loss(m, Xe, Ye), mlp_params=sum(q.numel() for q in m.parameters()))
        print("mlp", n, row["mlp_train"], row["mlp_test"], f"{time.time()-t0:.0f}s", flush=True)
        if n == 8: best_model = m
    sweep.append(row)
D["sweep"] = sweep
D["mlp_sample"] = sample(best_model, chars, stoi, "\n" * 8, 240, seed=1)

# ---- 3. learned character embeddings of the n=8 model, PCA to 2-D ----
E = best_model.emb.weight.detach(); Ec = E - E.mean(0)
U, S, Vh = torch.linalg.svd(Ec, full_matrices=False)
xy = (Ec @ Vh[:2].T).numpy()
D["charemb"] = {"xy": xy.round(3).tolist(), "var_explained": [float(s) for s in (S[:2] ** 2 / (S ** 2).sum())]}

json.dump(D, open(os.path.join(HERE, "data.json"), "w"))
print("saved", os.path.getsize(os.path.join(HERE, "data.json")), "bytes; now run data_gpt2.py")
