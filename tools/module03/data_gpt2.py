"""Extract the GPT-2 embedding data for module 3 (updates data.json in place; run data.py first)."""
import os, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))
# ---- 4. GPT-2 token embedding table (wte) ----
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2"); gm = GPT2LMHeadModel.from_pretrained("gpt2")
Wg = gm.transformer.wte.weight.detach(); Wn = Wg / Wg.norm(dim=1, keepdim=True)
def tid(s):
    t = tok.encode(s); assert len(t) == 1, (s, t); return t[0]
G = {"shape": list(Wg.shape), "tied": gm.lm_head.weight.data_ptr() == gm.transformer.wte.weight.data_ptr(),
     "norm_mean": float(Wg.norm(dim=1).mean()), "mean_vec_norm": float(Wg.mean(0).norm())}
gen = torch.Generator().manual_seed(0); pr = torch.randint(0, Wg.shape[0], (20000, 2), generator=gen)
G["random_pair_cos"] = float((Wn[pr[:, 0]] * Wn[pr[:, 1]]).sum(1).mean())
G["lookup"] = []
for s in ("The", " cat", " sat"):
    i = tid(s); G["lookup"].append({"tok": s, "id": i, "row": Wg[i, :5].numpy().round(3).tolist()})
def neighbours(s, k=6):
    sims = Wn @ Wn[tid(s)]; top = sims.topk(k + 1).indices[1:]
    return [[tok.decode([int(i)]), round(float(sims[i]), 3)] for i in top]
G["nn"] = {s: neighbours(s) for s in (" king", " Paris", " Monday", " 1999", " dog", " happy")}
def analogy(a, b, c, k=5):
    q = Wn[tid(a)] - Wn[tid(b)] + Wn[tid(c)]; q = q / q.norm(); sims = Wn @ q
    for s in (a, b, c): sims[tid(s)] = -9
    return [tok.decode([int(i)]) for i in sims.topk(k).indices]
G["analogy"] = [[a, b, c, analogy(a, b, c)] for a, b, c in
                [(" king", " man", " woman"), (" Paris", " France", " Italy"), (" walking", " walk", " swim"), (" Rome", " Italy", " Japan")]]
groups = {"countries": [" France", " Germany", " Italy", " Spain", " Japan", " China", " Russia", " Egypt"],
          "capitals": [" Paris", " Berlin", " Rome", " Madrid", " Tokyo", " Beijing", " Moscow", " Cairo"]}
allt = [(g, s) for g, ss in groups.items() for s in ss]
M = Wg[[tid(s) for _, s in allt]]; Mc = M - M.mean(0)
U, S, Vh = torch.linalg.svd(Mc, full_matrices=False)
G["groups"] = {"points": [[g, s.strip(), *(Mc @ Vh[:2].T)[i].numpy().round(3).tolist()] for i, (g, s) in enumerate(allt)],
               "var_explained": [float(s) for s in (S[:2] ** 2 / (S ** 2).sum())]}
D["gpt2"] = G
json.dump(D, open(os.path.join(HERE, "data.json"), "w"))
print("saved", os.path.getsize(os.path.join(HERE, "data.json")), "bytes")
