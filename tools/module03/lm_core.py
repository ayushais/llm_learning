# --- shared by data.py, figs.py and the notebook (mknb.py pastes this file into the notebook) ---
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def make_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi

# ---------- count-based n-gram (the classical baseline) ----------
def ngram_keys(ids, n, V):
    """For every position t >= n: integer code of the n previous symbols (ctx) and of ctx+next (key)."""
    ids = np.asarray(ids, dtype=np.int64)
    ctx = np.zeros(len(ids) - n, dtype=np.int64)
    for k in range(n):                       # base-V number built from ids[t-n .. t-1]
        ctx = ctx * V + ids[k : len(ids) - n + k]
    return ctx, ctx * V + ids[n:]

class CountNgram:
    """P(x | previous n symbols) = (count(ctx, x) + a) / (count(ctx) + a V). Stored sparsely."""
    def __init__(self, ids, n, V):
        self.n, self.V = n, V
        ctx, key = ngram_keys(ids, n, V)
        self.ctx_u, self.ctx_c = np.unique(ctx, return_counts=True)
        self.key_u, self.key_c = np.unique(key, return_counts=True)
    @staticmethod
    def _look(u, c, q):
        i = np.searchsorted(u, q); i[i == len(u)] = 0
        return np.where(u[i] == q, c[i], 0)
    def nll(self, ids, a):
        """mean negative log-likelihood (nats/token) of ids with add-a smoothing"""
        ctx, key = ngram_keys(ids, self.n, self.V)
        cc = self._look(self.ctx_u, self.ctx_c, ctx); ck = self._look(self.key_u, self.key_c, key)
        return float(-np.log((ck + a) / (cc + a * self.V)).mean())

# ---------- Bengio-style neural LM: embed n previous symbols, concat, tanh MLP, softmax ----------
class NeuralLM(nn.Module):
    def __init__(self, V, n, d=16, h=256):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(V, d)          # the lookup table C of Bengio et al. 2003
        self.hid = nn.Linear(n * d, h)
        self.out = nn.Linear(h, V)
    def forward(self, x):                      # x: (B, n) ints -> logits (B, V)
        e = self.emb(x).flatten(1)             # (B, n*d): concatenated embeddings
        return self.out(torch.tanh(self.hid(e)))

def windows(ids, n):
    ids = torch.as_tensor(ids, dtype=torch.long)
    idx = torch.arange(n)[None, :] + torch.arange(len(ids) - n)[:, None]
    return ids[idx], ids[n:]

@torch.no_grad()
def eval_loss(model, X, Y, bs=20000):
    return float(sum(F.cross_entropy(model(X[i:i+bs]), Y[i:i+bs], reduction="sum") for i in range(0, len(X), bs)) / len(X))

def train_lm(model, X, Y, steps=6000, bs=256, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(0, len(X), (bs,), generator=g)
        loss = F.cross_entropy(model(X[i]), Y[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

@torch.no_grad()
def sample(model, chars, stoi, prompt, n_new, seed=0, temperature=1.0):
    g = torch.Generator().manual_seed(seed)
    ids = [stoi[c] for c in prompt]
    for _ in range(n_new):
        ctx = ids[-model.n:]
        ctx = [0] * (model.n - len(ctx)) + ctx
        p = F.softmax(model(torch.tensor([ctx])) / temperature, -1)[0]
        ids.append(int(torch.multinomial(p, 1, generator=g)))
    return "".join(chars[i] for i in ids)
