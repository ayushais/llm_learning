"""Extract the real data used by the module 5 figures (run once; writes data.json). About 25 minutes on a laptop CPU."""
import os, sys, json, math, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_core import *
torch.manual_seed(0); np.random.seed(0)
D = {}
log = lambda *a: print(*a, flush=True)

# ---- 1. activation functions (computed curves, for the FFN figure) ----
z = torch.linspace(-4, 3, 141)
D["act"] = {"z": z.tolist(), "relu": F.relu(z).tolist(), "gelu": F.gelu(z).tolist(), "silu": F.silu(z).tolist()}

# ---- 2. GPT-2 small: our block vs Hugging Face, the residual stream, logit lens, massive activations, LN -> RMSNorm ----
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
gm = GPT2LMHeadModel.from_pretrained("gpt2").eval()
text = open(os.path.join(HERE, "..", "..", "data", "tinyshakespeare.txt")).read()
enc = tok.encode(text[:100000]); T = 256
x = torch.tensor([enc[i * T:(i + 1) * T] for i in range(8)])            # 8 windows x 256 tokens
with torch.no_grad():
    o = gm(x, output_hidden_states=True)
    # hidden_states[12] in Hugging Face is already after the final LayerNorm; rebuild the raw stream layer by layer
    H = [o.hidden_states[0]]; upd = []
    for blk in gm.transformer.h:
        h0 = H[-1]; a = blk.attn(blk.ln_1(h0))[0]; h1 = h0 + a; m = blk.mlp(blk.ln_2(h1)); H.append(h1 + m)
        upd.append({"attn": float((a[:, 1:].norm(dim=-1) / h0[:, 1:].norm(dim=-1)).mean()),
                    "mlp": float((m[:, 1:].norm(dim=-1) / h1[:, 1:].norm(dim=-1)).mean())})
    logits = gm.lm_head(gm.transformer.ln_f(H[-1]))
    assert torch.allclose(logits, o.logits, atol=1e-3)

    # 2a. one GPT-2 block from scratch (our LayerNorm, our MLP, Module 4 attention) against the Hugging Face block
    l = 5; hb = gm.transformer.h[l]; d = 768
    ln1, ln2 = LayerNorm(d), LayerNorm(d); mlp = MLP(d); att = MultiHeadAttention(d, 12)
    for ours, theirs in ((ln1, hb.ln_1), (ln2, hb.ln_2)): ours.weight.copy_(theirs.weight); ours.bias.copy_(theirs.bias)
    att.qkv.weight.copy_(hb.attn.c_attn.weight.T); att.qkv.bias.copy_(hb.attn.c_attn.bias)     # GPT-2's Conv1D stores (in, out)
    att.proj.weight.copy_(hb.attn.c_proj.weight.T); att.proj.bias.copy_(hb.attn.c_proj.bias)
    mlp.fc.weight.copy_(hb.mlp.c_fc.weight.T); mlp.fc.bias.copy_(hb.mlp.c_fc.bias)
    mlp.proj.weight.copy_(hb.mlp.c_proj.weight.T); mlp.proj.bias.copy_(hb.mlp.c_proj.bias)
    hh = H[l] + att(ln1(H[l])); hh = hh + mlp(ln2(hh))
    D["gpt2_check"] = {"layer": l, "max_abs_diff": float((hh - H[l + 1]).abs().max()), "stream_scale": float(H[l + 1].abs().max())}

    y = x[:, 1:]
    base = float(F.cross_entropy(logits[:, :-1].flatten(0, 1), y.flatten()))
    top_final = logits[:, :-1].argmax(-1)
    lens = []
    for l in range(13):
        lg = gm.lm_head(gm.transformer.ln_f(H[l]))[:, :-1]
        lens.append({"loss": float(F.cross_entropy(lg.flatten(0, 1), y.flatten())),
                     "agree": float((lg.argmax(-1) == top_final).float().mean()),
                     "acc": float((lg.argmax(-1) == y).float().mean())})
    rest = [float(h[:, 1:].norm(dim=-1).mean()) for h in H]
    first = [float(h[:, 0].norm(dim=-1).mean()) for h in H]
    mean_over_rms = [float((h[:, 1:].mean(-1).abs() / h[:, 1:].pow(2).mean(-1).sqrt()).mean()) for h in H]
    mid = H[6][:, 0]                                                       # token 0 in the middle of the network
    top = mid.abs().mean(0).topk(3)
    other = H[6][:, 1:].abs()
    D["stream"] = {"norm_rest": rest, "norm_first": first, "upd": upd, "mean_over_rms": mean_over_rms,
                   "massive_dims": top.indices.tolist(), "massive_vals": [float(v) for v in mid.mean(0)[top.indices]],
                   "other_typical_abs": float(other.median()), "other_max_same_dims": float(other[..., top.indices].abs().max()),
                   "first_token_text": tok.decode([enc[0]])}
    D["lens"] = {"layers": lens, "loss": base, "uniform": math.log(50257)}

    # 2b. replace every LayerNorm of GPT-2 by an RMSNorm (same gain and bias, no mean subtraction): how much does the loss move?
    def rms_forward(mod):
        def f(h): return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + mod.eps) * mod.weight + mod.bias
        return f
    lns = [m for m in gm.modules() if isinstance(m, torch.nn.LayerNorm)]
    saved = [m.forward for m in lns]
    for m_ in lns: m_.forward = rms_forward(m_)
    lg2 = gm(x).logits
    for m_, f in zip(lns, saved): m_.forward = f
    D["ln_swap"] = {"n_norms": len(lns), "loss_ln": base, "loss_rms": float(F.cross_entropy(lg2[:, :-1].flatten(0, 1), y.flatten())),
                    "top1_agree": float((lg2[:, :-1].argmax(-1) == top_final).float().mean())}
log(json.dumps({k: D[k] for k in ("gpt2_check", "ln_swap")}), D["stream"]["massive_dims"], D["stream"]["massive_vals"])

# 2c. speed of the two norms on CPU (our code and PyTorch's), d = 768, 64 x 256 tokens
xx = torch.randn(64, 256, 768)
def bench(f, n=30):
    for _ in range(3): f(xx)
    t = time.perf_counter()
    for _ in range(n): f(xx)
    return (time.perf_counter() - t) / n * 1000
with torch.no_grad():
    D["speed_ms"] = {"ours_ln": bench(LayerNorm(768)), "ours_rms": bench(RMSNorm(768)),
                     "torch_ln": bench(torch.nn.LayerNorm(768)), "torch_rms": bench(torch.nn.RMSNorm(768))}
log(D["speed_ms"])

# ---- 3. tiny character models on TinyShakespeare ----
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
ids = np.array([stoi[c] for c in text]); cut = int(0.9 * len(ids)); tr, te = ids[:cut], ids[cut:]

# 3a. gradient norm per block at initialisation, 12 blocks
DEPTH = dict(V=V, T=32, d=64, h=4, L=12)
VARIANTS = {"pre-LN": dict(norm="ln", place="pre", ffn="gelu"), "post-LN": dict(norm="ln", place="post", ffn="gelu"),
            "no residual": dict(norm="ln", place="pre", ffn="gelu", residual=False)}
def deep(kw, seed=0):
    torch.manual_seed(seed); return GPT(DEPTH["V"], DEPTH["T"], DEPTH["d"], DEPTH["h"], DEPTH["L"], **kw)
D["grad_init"] = {n: grad_norm_per_block(deep(kw), tr) for n, kw in VARIANTS.items()}
D["cos_init"] = {n: token_similarity_per_block(deep(kw), tr) for n, kw in VARIANTS.items()}

# 3b. training curves at a high learning rate, and a learning-rate sweep (pre vs post)
STEPS = 400
D["deep_curves"] = {}
for n, kw in VARIANTS.items():
    t0 = time.time(); D["deep_curves"][n] = train(deep(kw), tr, steps=STEPS, lr=1e-2, bs=32, log_every=20); log(n, D["deep_curves"][n][-1], f"{time.time()-t0:.0f}s")
m_ = deep(VARIANTS["post-LN"]); D["deep_curves"]["post-LN + warmup"] = train(m_, tr, steps=STEPS, lr=1e-2, bs=32, warmup=100, log_every=20)
D["lr_sweep"] = {"lrs": [1e-3, 3e-3, 1e-2, 3e-2]}
for n in ("pre-LN", "post-LN"):
    D["lr_sweep"][n] = [train(deep(VARIANTS[n]), tr, steps=STEPS, lr=lr, bs=32, log_every=20)[-1] for lr in D["lr_sweep"]["lrs"]]
    log(n, D["lr_sweep"][n])
D["unigram"] = float(-(np.bincount(tr, minlength=V) / len(tr) * np.log(np.bincount(tr, minlength=V) / len(tr) + 1e-12)).sum())

# 3c. ablation: from the Module 4 attention-only model to a full block, 2 layers, d = 64, two seeds
ABL = {"attention only (Module 4)": dict(norm=None, ffn=None), "+ GELU MLP": dict(norm=None, ffn="gelu"),
       "+ pre-LayerNorm (GPT-2 block)": dict(norm="ln", ffn="gelu"), "RMSNorm + SwiGLU (Llama block)": dict(norm="rms", ffn="swiglu")}
runs = {}
for n, kw in ABL.items():
    runs[n] = {"test": [], "train": []}
    for seed in (0, 1):
        torch.manual_seed(seed); m = GPT(V, 64, 64, 4, 2, **kw); t0 = time.time()
        train(m, tr, steps=2000, seed=seed)
        runs[n]["test"].append(evaluate(m, te)); runs[n]["train"].append(evaluate(m, tr[:len(te)]))
        runs[n]["params"] = sum(p.numel() for p in m.parameters())
        if seed == 0 and n.startswith("RMSNorm"): D["sample"] = sample(m, chars, stoi, "\n", 240, seed=1)
        log(n, seed, runs[n]["test"][-1], f"{time.time()-t0:.0f}s")
D["runs"] = runs
m3 = json.load(open(os.path.join(HERE, "..", "module03", "data.json")))
D["baselines"] = {"bigram": m3["bigram"]["gd_test"], "ngram4": [r for r in m3["sweep"] if r["n"] == 4][0]["ng_test"]}
json.dump(D, open(os.path.join(HERE, "data.json"), "w"))
log(json.dumps({k: D[k] for k in ("grad_init", "lr_sweep", "runs", "unigram")}, indent=1))
log(D["sample"])
