"""Build notebooks/03_embeddings_bigram.ipynb. The model code is read from lm_core.py,
so the notebook, the figures and the data script all use the same implementation."""
import os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
core = open(os.path.join(HERE, "lm_core.py")).read()
nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M("""# Module 3: Embeddings and a bigram baseline
Companion to `docs/03_embeddings_bigram.html`. We (1) show that an embedding layer is a table lookup, (2) build a character-level
**bigram** language model two ways, by counting and by gradient descent, and check that they agree, (3) compare count-based n-grams with a
**Bengio-style neural language model** (Bengio et al. 2003), and (4) look inside GPT-2's real embedding table.
Runs on a laptop CPU in about 7 minutes (the GPT-2 download is the only network step after the text file).""")
C("""import os, math, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)

os.makedirs("data", exist_ok=True)
path = "data/tinyshakespeare.txt"
if not os.path.exists(path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = open(path).read()""")
M("""## 0. Model code
The count-based n-gram and the neural LM live in one block, pasted from `tools/module03/lm_core.py` so this notebook is standalone. We read it in the sections that use it.""")
C(core)
C("""chars, stoi = make_vocab(text); V = len(chars)
ids = np.array([stoi[c] for c in text])
cut = int(len(ids) * 0.9); tr, te = ids[:cut], ids[cut:]
print(f"{len(text):,} characters, vocabulary V = {V}; train {len(tr):,} / held-out {len(te):,}")
print("uniform guess = ln V =", round(math.log(V), 3), "nats/char  (perplexity", V, ")")""")

M("""## 1. An embedding layer is a table lookup
`E[x]` (what `nn.Embedding` does) equals `one_hot(x) @ E`. We check that, and that the gradient reaches only the rows that were used.""")
C("""d = 8
emb = nn.Embedding(V, d)
x = torch.tensor([3, 17, 17, 40])
lookup = emb(x)
onehot = F.one_hot(x, V).float() @ emb.weight          # same thing as a matrix product
print("lookup == one_hot @ E :", torch.allclose(lookup, onehot))

lookup.sum().backward()
rows_with_grad = (emb.weight.grad.abs().sum(1) > 0).nonzero().flatten().tolist()
print("rows that received a gradient:", rows_with_grad, "  (batch used ids", sorted(set(x.tolist())), ")")""")
M("Only rows 3, 17 and 40 are touched (17 appears twice, so its gradient is accumulated). The other 62 rows are unchanged by this step. This is why rare tokens end up with poorly trained embeddings.")

M("""## 2. Cross-entropy by hand
`F.cross_entropy(logits, y)` is the mean of `-log softmax(logits)[y]`. Write it out and compare.""")
C("""logits = torch.randn(5, V); y = torch.randint(0, V, (5,))
logp = logits - logits.logsumexp(1, keepdim=True)       # log softmax, written out
manual = -logp[torch.arange(5), y].mean()
print(float(manual), float(F.cross_entropy(logits, y)))
assert torch.allclose(manual, F.cross_entropy(logits, y))
print("all-zero logits give a uniform guess:", float(F.cross_entropy(torch.zeros(5, V), y)), "= ln V =", math.log(V))""")

M("""## 3. The bigram model: count it, or train it
A bigram model predicts the next character from the previous one: a `V x V` table of logits `B`, `P(next | prev=i) = softmax(B[i])`.

**By counting.** The maximum-likelihood table is the normalised pair counts.""")
C("""X, Y = torch.tensor(tr[:-1]), torch.tensor(tr[1:])
Xt, Yt = torch.tensor(te[:-1]), torch.tensor(te[1:])
cnt = torch.zeros(V, V); cnt.index_put_((X, Y), torch.ones(len(X)), accumulate=True)
P_mle = cnt / cnt.sum(1, keepdim=True)
print("train loss of the count table:", float(-torch.log(P_mle[X, Y]).mean()))
print("pairs never seen in training:", int((cnt == 0).sum()), "of", V * V)

# held-out loss needs smoothing, otherwise an unseen pair has probability 0 and loss infinity
P_sm = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True)
print("held-out loss (add-0.5 smoothing):", float(-torch.log(P_sm[Xt, Yt]).mean()))
uni = np.bincount(tr, minlength=V) / len(tr)
print("unigram (no context) held-out loss:", float(-np.log(uni[te]).mean()))""")
M("**By gradient descent.** Treat `B` as parameters, start from zeros (uniform), and minimise the cross-entropy. For each row the gradient is the model's predicted distribution minus the empirical one, so it vanishes exactly at the count solution.")
C("""B = torch.zeros(V, V, requires_grad=True)
opt = torch.optim.Adam([B], lr=0.1)
curve = []
for step in range(501):
    loss = F.cross_entropy(B[X], Y)                     # B[X] is an embedding lookup whose output is the logits
    if step % 25 == 0: curve.append((step, loss.item()))
    opt.zero_grad(); loss.backward(); opt.step()
mle = float(-torch.log(P_mle[X, Y]).mean())
print(f"after 500 steps: {curve[-1][1]:.4f}  vs count solution {mle:.4f}  (gap {curve[-1][1]-mle:.5f})")
print("largest difference between softmax(B) and the counted probabilities:", float((F.softmax(B.detach(), 1) - P_mle).abs().max()))
plt.figure(figsize=(5, 3)); plt.plot(*zip(*curve)); plt.axhline(mle, ls="--", c="C1"); plt.xlabel("step"); plt.ylabel("train loss"); plt.title("gradient descent reaches the count solution"); plt.show()""")
M("Gradient descent walks to the counting answer (rows of rare characters converge last, which is why the largest probability difference is not exactly zero). **Training a language model by gradient descent is a way of estimating conditional probabilities.** Now look at the table and sample from it.")
C("""top = sorted(range(V), key=lambda i: -uni[i])[:30]
lab = [repr(chars[i])[1:-1] for i in top]
plt.figure(figsize=(7, 6.5)); plt.imshow(P_mle[top][:, top].sqrt().numpy(), cmap="Blues")
plt.xticks(range(30), lab, fontsize=7); plt.yticks(range(30), lab, fontsize=7)
plt.xlabel("next character"); plt.ylabel("previous character"); plt.title("P(next | previous), sqrt scale, 30 most frequent characters"); plt.show()
for c in "t.\\n":
    i = stoi[c]; top4 = P_mle[i].topk(4)
    print(repr(c), "->", [(chars[j], round(float(p), 3)) for p, j in zip(top4.values, top4.indices)])""")
C("""class BigramLM(nn.Module):
    n = 1
    def __init__(s, B): super().__init__(); s.B = B
    def forward(s, x): return s.B[x[:, -1]]
print(sample(BigramLM(B.detach()), chars, stoi, "\\n", 240, seed=1))""")
M("Plausible letter pairs, almost no words: with one character of context the model cannot know which word it is in.")

M("""## 4. More context: count n-grams hit a wall, a neural LM does not
**Count-based n-gram** (class `CountNgram` above): condition on the previous `n` characters, `P = (count(ctx, x) + a) / (count(ctx) + a V)`. The table has `V^(n+1)` cells. We pick the smoothing `a` on a validation slice of the training data.""")
C("""tr_fit, va = tr[: int(len(tr) * 0.9)], tr[int(len(tr) * 0.9):]
ng_rows = []
for n in range(0, 7):
    ng = CountNgram(tr_fit, n, V)
    a = min((ng.nll(va, a), a) for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0))[1]
    ng = CountNgram(tr, n, V)
    ng_rows.append((n, ng.nll(tr, a), ng.nll(te, a), len(ng.key_u)))
    print(f"n={n}: a={a:<5} train {ng_rows[-1][1]:.3f}  held-out {ng_rows[-1][2]:.3f}  cells seen {len(ng.key_u):>9,} of {float(V)**(n+1):.2e}")""")
M("""The training loss keeps falling (the table memorises), but the held-out loss turns around: most long contexts were never seen.

**Neural LM (Bengio et al. 2003).** Look up the `n` previous characters in a shared embedding table, concatenate, one `tanh` hidden layer, softmax over the vocabulary (class `NeuralLM` above). The number of parameters grows linearly with `n`, not exponentially. Here we train for 4,000 steps at `n` = 1, 2, 4, 8 (the lesson's figure uses 6,000 steps at every `n` from 1 to 8, so the numbers differ slightly).""")
C("""nl_rows = []
models = {}
for n in (1, 2, 4, 8):
    Xn, Yn = windows(tr, n); Xe, Ye = windows(te, n)
    torch.manual_seed(0); m = NeuralLM(V, n); t0 = time.time(); train_lm(m, Xn, Yn, steps=4000)
    models[n] = m
    nl_rows.append((n, eval_loss(m, Xn, Yn), eval_loss(m, Xe, Ye), sum(p.numel() for p in m.parameters())))
    print(f"n={n}: train {nl_rows[-1][1]:.3f}  held-out {nl_rows[-1][2]:.3f}  params {nl_rows[-1][3]:,}  ({time.time()-t0:.0f}s)")""")
C("""plt.figure(figsize=(6, 3.5))
plt.plot([r[0] for r in ng_rows], [r[2] for r in ng_rows], "o-", c="C0", label="count n-gram, held-out")
plt.plot([r[0] for r in ng_rows], [r[1] for r in ng_rows], "o--", c="C0", label="count n-gram, train")
plt.plot([r[0] for r in nl_rows], [r[2] for r in nl_rows], "o-", c="C1", label="neural LM, held-out")
plt.plot([r[0] for r in nl_rows], [r[1] for r in nl_rows], "o--", c="C1", label="neural LM, train")
plt.xlabel("context length n"); plt.ylabel("nats / char"); plt.legend(); plt.show()
print(sample(models[8], chars, stoi, "\\n" * 8, 240, seed=1).lstrip("\\n"))""")
M("""Read the plot honestly: at short contexts (n = 3, 4) the plain count model is as good or better than our small, briefly trained neural model; the neural model's advantage is that it does not collapse as the context grows. Neither can use a long context well, which is what attention (module 4) fixes.

## 5. Inside GPT-2's embedding table
GPT-2 small's `wte` is the same kind of object, `50,257 x 768`, trained inside a deep transformer.""")
C("""from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2"); gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
Wg = gpt2.transformer.wte.weight.detach()
print("wte shape:", tuple(Wg.shape), "=", f"{Wg.numel():,}", "parameters;", f"{Wg.numel() / sum(p.numel() for p in gpt2.parameters()):.1%} of the model")
print("input and output embeddings are the same tensor (weight tying):", gpt2.lm_head.weight.data_ptr() == gpt2.transformer.wte.weight.data_ptr())
ids_ = tok.encode("The cat sat")
print(ids_, [tok.decode([i]) for i in ids_]); print(Wg[ids_][:, :5])""")
C("""Wn = Wg / Wg.norm(dim=1, keepdim=True)
def tid(s):
    t = tok.encode(s); assert len(t) == 1, (s, t); return t[0]
def neighbours(s, k=6):
    sims = Wn @ Wn[tid(s)]; top = sims.topk(k + 1).indices[1:]
    return [(tok.decode([int(i)]), round(float(sims[i]), 2)) for i in top]
for s in (" king", " Paris", " Monday", " 1999", " dog", " happy"):
    print(repr(s), neighbours(s))""")
M("The neighbours are spelling variants (with or without a leading space, capitalised or not) and genuine semantic relatives. Now the analogy test of Mikolov et al. (the query words are excluded from the answers, as is standard).")
C("""def analogy(a, b, c, k=5):
    q = Wn[tid(a)] - Wn[tid(b)] + Wn[tid(c)]; q = q / q.norm(); sims = Wn @ q
    for s in (a, b, c): sims[tid(s)] = -9
    return [tok.decode([int(i)]) for i in sims.topk(k).indices]
for a, b, c in [(" king", " man", " woman"), (" Paris", " France", " Italy"), (" walking", " walk", " swim"), (" Rome", " Italy", " Japan")]:
    print(f"{a} - {b} + {c} ->", analogy(a, b, c))""")
M("Some work, some do not (for Paris - France + Italy the nearest row is `Paris` itself and `Rome` is only fifth). The structure is real but approximate. One more caution: cosine similarity has a large common offset.")
C("""gen = torch.Generator().manual_seed(0); pr = torch.randint(0, Wg.shape[0], (20000, 2), generator=gen)
print("mean cosine between two random rows:", float((Wn[pr[:, 0]] * Wn[pr[:, 1]]).sum(1).mean()))
print("norm of the mean row:", float(Wg.mean(0).norm()), " average row norm:", float(Wg.norm(dim=1).mean()))""")
M("""Finally, a 2-D look at three kinds of words (PCA fitted on just these 23 tokens). Nobody told the model the groups, yet each kind forms its own cluster.""")
C("""groups = {"weekdays": [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"],
          "countries": [" France", " Germany", " Italy", " Spain", " Japan", " China", " Russia", " Egypt"],
          "animals": [" dog", " cat", " horse", " cow", " pig", " sheep", " lion", " tiger"]}
words = [(g, w) for g, ws in groups.items() for w in ws]
Mx = Wg[[tid(w) for _, w in words]]; Mc = Mx - Mx.mean(0)
_, S, Vh = torch.linalg.svd(Mc, full_matrices=False); xy = (Mc @ Vh[:2].T).numpy()
plt.figure(figsize=(6.5, 4.5))
for j, g in enumerate(groups):
    idx = [i for i, (gg, _) in enumerate(words) if gg == g]
    plt.scatter(*xy[idx].T, c=f"C{j}", label=g)
plt.legend(); plt.title(f"GPT-2 wte, PCA of {len(words)} tokens ({(S[:2]**2).sum() / (S**2).sum():.0%} of variance)"); plt.show()""")

M("""## Exercises
1. **Embedding gradient.** Make a batch whose tokens are all id 5 and check the gradient with `nn.Embedding` is `batch_size` times the gradient for a single occurrence. What does that imply for very frequent tokens?
2. **Smoothing.** The bigram model needs smoothing for held-out data. Try `a` in `(0.01, 0.1, 0.5, 1, 5)` and plot held-out loss against `a`. Why does the best value shrink as the context `n` grows in section 4? (Look at the `a` the validation slice picked.)
3. **Tying.** Change `NeuralLM` so the output layer reuses the embedding matrix (this needs `d = h`, or an extra projection). Does held-out loss at `n = 4` change? How many parameters did you save?
4. **Open-ended: what do the character embeddings learn?** Take `models[8].emb.weight`, project to 2-D with PCA, and label the points. Do vowels group? Capitals vs lowercase? Punctuation? Repeat with embedding sizes `d = 2, 4, 16, 64` and compare held-out loss. Does a 2-D embedding still work?
5. **Open-ended: other languages.** Run the neural LM on another text file (a different language or code). What changes in the learned embeddings?
6. **GPT-2.** Find 5 tokens whose nearest neighbours are *not* semantic (for instance, fragments or odd symbols). What do the neighbours tell you about the data the model saw?""")
nbf.write(nb, os.path.join(HERE, "..", "..", "notebooks", "03_embeddings_bigram.ipynb"))
print("cells:", len(nb.cells))
