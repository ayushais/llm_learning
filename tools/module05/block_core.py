# --- shared by data.py, figs.py and the notebook (mknb.py pastes this file into the notebook) ---
import math, torch, torch.nn as nn, torch.nn.functional as F

def attention(q, k, v):
    """Causal scaled dot-product attention (Module 4). q, k, v: (..., T, d)."""
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return F.softmax(s, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, bias=True):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=bias)
        self.proj = nn.Linear(d, d, bias=bias)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (z.view(B, T, self.h, D // self.h).transpose(1, 2) for z in self.qkv(x).split(D, -1))
        return self.proj(attention(q, k, v).transpose(1, 2).reshape(B, T, D))

class LayerNorm(nn.Module):
    """Ba et al. 2016: subtract the mean, divide by the standard deviation, then a learned gain and bias."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight, self.bias = eps, nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)              # biased variance, as in PyTorch
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias

class RMSNorm(nn.Module):
    """Zhang & Sennrich 2019: no mean subtraction, no bias; divide by the root mean square, then a learned gain."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MLP(nn.Module):
    """The original position-wise feed-forward network: d -> 4d -> d with a GELU (GPT-2 uses the tanh approximation)."""
    def __init__(self, d, hidden=None, bias=True, approximate="tanh"):
        super().__init__()
        hidden = hidden or 4 * d
        self.fc, self.proj, self.approximate = nn.Linear(d, hidden, bias=bias), nn.Linear(hidden, d, bias=bias), approximate
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate=self.approximate))

class SwiGLU(nn.Module):
    """Shazeer 2020: (Swish(x W_gate) * x W_up) W_down, no biases. hidden = 2/3 * 4d keeps the parameter count of the 4d MLP."""
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or int(2 * 4 * d / 3)
        self.gate, self.up, self.down = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))     # silu(z) = z * sigmoid(z) = Swish with beta = 1

NORMS = {"ln": LayerNorm, "rms": RMSNorm, None: lambda d: nn.Identity()}

class Block(nn.Module):
    """One transformer block. norm: 'ln' | 'rms' | None;  place: 'pre' | 'post';  ffn: 'gelu' | 'swiglu' | None;  residual: bool."""
    def __init__(self, d, h, norm="rms", place="pre", ffn="swiglu", residual=True):
        super().__init__()
        self.place, self.residual = place, residual
        self.n1, self.att = NORMS[norm](d), MultiHeadAttention(d, h)
        self.ffn = {"gelu": lambda: MLP(d), "swiglu": lambda: SwiGLU(d), None: lambda: None}[ffn]()
        self.n2 = NORMS[norm](d) if self.ffn is not None else None
    def forward(self, x):
        r = 1.0 if self.residual else 0.0                        # residual=False: same block, the "x +" removed
        if self.place == "pre":                                  # x + f(norm(x)): the stream itself is never normalised
            x = r * x + self.att(self.n1(x))
            if self.ffn is not None: x = r * x + self.ffn(self.n2(x))
        else:                                                    # post-LN (Vaswani 2017): norm(x + f(x))
            x = self.n1(r * x + self.att(x))
            if self.ffn is not None: x = self.n2(r * x + self.ffn(x))
        return x

class GPT(nn.Module):
    """Character LM: token + learned position embeddings, L blocks, a final norm (pre-norm models only), linear head."""
    def __init__(self, V, T, d=64, h=4, L=2, **blk):
        super().__init__()
        self.T = T
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h, **blk) for _ in range(L))
        pre_norm = blk.get("place", "pre") == "pre" and blk.get("norm", "rms") is not None
        self.nf = NORMS[blk.get("norm", "rms")](d) if pre_norm else nn.Identity()
        self.out = nn.Linear(d, V)
    def forward(self, x):
        z = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks: z = b(z)
        return self.out(self.nf(z))

def batches(ids, T, bs, g):
    ids = torch.as_tensor(ids, dtype=torch.long)
    i = torch.randint(0, len(ids) - T - 1, (bs,), generator=g)
    idx = i[:, None] + torch.arange(T + 1)[None, :]
    return ids[idx][:, :-1], ids[idx][:, 1:]

def train(model, ids, steps=3000, bs=64, lr=3e-3, warmup=0, seed=0, log_every=50):
    """Adam, optional linear warmup, then cosine decay. Returns the training loss averaged over each log_every steps (nan once diverged)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_at = lambda s: lr * (s + 1) / warmup if s < warmup else lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    hist, acc = [], 0.0
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr_at(s)
        x, y = batches(ids, model.T, bs, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            hist += [float("nan")] * ((steps - s) // log_every); break
        opt.zero_grad(); loss.backward(); opt.step()
        acc += loss.item()
        if (s + 1) % log_every == 0: hist.append(acc / log_every); acc = 0.0
    return hist

@torch.no_grad()
def evaluate(model, ids, min_pos=8):
    """Mean loss over non-overlapping windows, scoring positions >= min_pos (same protocol as Module 4)."""
    ids = torch.as_tensor(ids, dtype=torch.long); T = model.T; n = (len(ids) - 1) // T
    x = ids[: n * T].view(n, T); y = ids[1 : n * T + 1].view(n, T)
    tot = 0.0
    for i in range(0, n, 256):
        lg = model(x[i:i+256])[:, min_pos:]
        tot += float(F.cross_entropy(lg.flatten(0, 1), y[i:i+256, min_pos:].flatten(), reduction="sum"))
    return tot / (n * (T - min_pos))

def grad_norm_per_block(model, ids, bs=64, seed=0):
    """One backward pass at the current weights; the gradient norm of each block's parameters."""
    x, y = batches(ids, model.T, bs, torch.Generator().manual_seed(seed))
    model.zero_grad(); F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
    return [math.sqrt(sum(float(p.grad.pow(2).sum()) for p in b.parameters())) for b in model.blocks]

@torch.no_grad()
def token_similarity_per_block(model, ids, bs=64, seed=0):
    """Mean cosine similarity between the vectors of different positions of a sequence, after each block (rank collapse if it goes to 1)."""
    x, _ = batches(ids, model.T, bs, torch.Generator().manual_seed(seed))
    z = model.tok(x) + model.pos(torch.arange(x.shape[1])); out = []
    T = x.shape[1]; off = ~torch.eye(T, dtype=torch.bool)
    for b in model.blocks:
        z = b(z); u = F.normalize(z, dim=-1)
        out.append(float((u @ u.transpose(1, 2))[:, off].mean()))
    return out

@torch.no_grad()
def sample(model, chars, stoi, prompt, n_new, seed=0):
    g = torch.Generator().manual_seed(seed); ids = [stoi[c] for c in prompt]
    for _ in range(n_new):
        p = F.softmax(model(torch.tensor([ids[-model.T:]]))[0, -1], -1)
        ids.append(int(torch.multinomial(p, 1, generator=g)))
    return "".join(chars[i] for i in ids)
