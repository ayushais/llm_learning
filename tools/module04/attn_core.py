# --- shared by data.py, figs.py and the notebook (mknb.py pastes this file into the notebook) ---
import math, torch, torch.nn as nn, torch.nn.functional as F

def attention(q, k, v, causal=True, scale=True):
    """Scaled dot-product attention. q, k, v: (..., T, d). Returns (output, weights)."""
    s = q @ k.transpose(-2, -1)                                   # (..., T, T): score[i, j] = q_i . k_j
    if scale: s = s / math.sqrt(q.shape[-1])
    if causal:
        T = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))   # hide the future
    w = F.softmax(s, dim=-1)                                      # each row sums to 1
    return w @ v, w

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)               # W_q, W_k, W_v side by side
        self.proj = nn.Linear(d_model, d_model)                   # W_o
    def forward(self, x, return_weights=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (z.view(B, T, self.h, self.dk).transpose(1, 2) for z in (q, k, v))      # (B, h, T, dk)
        y, w = attention(q, k, v)
        y = self.proj(y.transpose(1, 2).reshape(B, T, D))         # concatenate heads, mix with W_o
        return (y, w) if return_weights else y

class AttnLM(nn.Module):
    """Character LM: token + learned position embeddings, L attention layers with residuals, no MLP, no norm."""
    def __init__(self, V, T, d=64, h=4, L=1):
        super().__init__()
        self.T = T
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.att = nn.ModuleList(MultiHeadAttention(d, h) for _ in range(L))
        self.out = nn.Linear(d, V)
    def forward(self, x, return_weights=False):
        z = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        ws = []
        for a in self.att:
            y, w = a(z, True); z = z + y; ws.append(w)
        logits = self.out(z)
        return (logits, ws) if return_weights else logits

def batches(ids, T, bs, g):
    ids = torch.as_tensor(ids, dtype=torch.long)
    i = torch.randint(0, len(ids) - T - 1, (bs,), generator=g)
    idx = i[:, None] + torch.arange(T + 1)[None, :]
    return ids[idx][:, :-1], ids[idx][:, 1:]

def train_attn(model, ids, steps=3000, bs=64, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        x, y = batches(ids, model.T, bs, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

@torch.no_grad()
def eval_attn(model, ids, min_pos=8):
    """Mean loss over non-overlapping windows, scoring only positions >= min_pos (so every prediction sees at least min_pos characters)."""
    ids = torch.as_tensor(ids, dtype=torch.long); T = model.T; n = (len(ids) - 1) // T
    x = ids[: n * T].view(n, T); y = ids[1 : n * T + 1].view(n, T)
    tot = 0.0
    for i in range(0, n, 256):
        lg = model(x[i:i+256])[:, min_pos:]
        tot += float(F.cross_entropy(lg.flatten(0, 1), y[i:i+256, min_pos:].flatten(), reduction="sum"))
    return tot / (n * (T - min_pos))

@torch.no_grad()
def sample_attn(model, chars, stoi, prompt, n_new, seed=0):
    g = torch.Generator().manual_seed(seed); ids = [stoi[c] for c in prompt]
    for _ in range(n_new):
        ctx = torch.tensor([ids[-model.T:]])
        p = F.softmax(model(ctx)[0, -1], -1)
        ids.append(int(torch.multinomial(p, 1, generator=g)))
    return "".join(chars[i] for i in ids)
