# llm_learning

A self-paced course on how LLMs work, built on the original research: concepts, runnable code, and the tricks used in modern models.

- `docs/` : HTML lessons (self-contained, open `docs/index.html`)
- `notebooks/` : one standalone notebook per module, runnable on a CPU laptop
- `capstone/` : the final GPT, assembled from the modules

## Setup

```
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

## Modules

1. Big picture (done)
2. Tokenization and BPE
3. Embeddings and a bigram baseline
4. Self-attention step by step
5. The transformer block: residuals, RMSNorm, SwiGLU
6. Training: AdamW, schedules, scaling laws
7. Positional encodings: sinusoidal to RoPE
8. Efficient attention: KV cache, GQA, FlashAttention
9. Sampling and decoding
10. Quantization
11. LoRA fine-tuning
12. Alignment: RLHF and DPO
13. Reasoning and RL
14. Capstone
