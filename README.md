# llm_learning

A self-paced course on how large language models work, grounded in the original research. Each module pairs an illustrated HTML lesson with a runnable notebook, and the pieces come together in a small modern GPT you train yourself.

- **Read:** https://ayushais.github.io/llm_learning/ (works on a tablet; each lesson is a single self-contained HTML file)
- **Code:** the notebooks in `notebooks/`, runnable on a CPU-only laptop

## Goals

- Understand the whole pipeline: tokens, transformer, training, alignment, inference.
- Learn the "tricks" that make modern LLMs work (RoPE, GQA, KV cache, FlashAttention, quantization, LoRA, DPO, ...), each tied to the paper it came from.
- Build intuition by experiment: every concept has code you can run and change.

Prerequisites: Python, PyTorch basics, linear algebra, backpropagation.

## Repository layout

```
docs/          HTML lessons: index.html + NN_name.html (published via GitHub Pages)
notebooks/     One standalone notebook per module: NN_name.ipynb
capstone/      The final GPT, assembled from the modules
tools/         Scripts that generate lesson figures from real model data
.devcontainer/ GitHub Codespaces / VS Code dev container (installs CPU PyTorch + requirements)
requirements.txt
```

Lessons and notebooks share the number and name, e.g. `docs/01_big_picture.html` and `notebooks/01_big_picture.ipynb`.

## What a module contains

**Lesson (`docs/NN_name.html`)**
1. Motivation: what problem this solves, in a few sentences.
2. Core idea, with the math written out and every symbol explained.
3. Figures: diagrams of the mechanism, plus charts from real model data where possible. Anything schematic is labelled as such.
4. Details and variants as the papers present them, with citations (arXiv or official links).
5. Common misconceptions.
6. Reading list: the primary papers first, then good secondary explanations.
7. Check-yourself questions.

**Notebook (`notebooks/NN_name.ipynb`)**
- Standalone: it runs top to bottom without other notebooks, on CPU, in minutes.
- Explanation in markdown cells, code in short cells, outputs saved.
- Implements the idea from scratch first, then compares with a reference implementation (PyTorch, Hugging Face) where one exists.
- Ends with exercises, including at least one open-ended experiment.

**Capstone** reuses the from-scratch components (tokenizer, RoPE, RMSNorm, SwiGLU, GQA, KV cache) in one small GPT trained on a CPU.

## Modules

| # | Module | Key sources |
|---|--------|-------------|
| 1 | Big picture: pipeline, training stages, generation (done, [Colab](https://colab.research.google.com/github/ayushais/llm_learning/blob/main/notebooks/01_big_picture.ipynb)) | Vaswani 2017; Radford 2019 (GPT-2); Ouyang 2022 |
| 2 | Tokenization and BPE (done, [Colab](https://colab.research.google.com/github/ayushais/llm_learning/blob/main/notebooks/02_tokenization_bpe.ipynb)) | Sennrich 2016; GPT-2 (Radford 2019); Kudo & Richardson 2018; Petrov 2023 |
| 3 | Embeddings and a bigram baseline | Bengio 2003 |
| 4 | Self-attention step by step | Vaswani 2017 |
| 5 | The transformer block: residuals, RMSNorm, SwiGLU | Zhang & Sennrich 2019; Shazeer 2020 |
| 6 | Training: AdamW, schedules, mixed precision, scaling laws | Loshchilov & Hutter 2017; Kaplan 2020; Hoffmann 2022 |
| 7 | Positional encodings: sinusoidal to RoPE | Su 2021 |
| 8 | Efficient attention: KV cache, MQA/GQA, FlashAttention | Shazeer 2019; Ainslie 2023; Dao 2022 |
| 9 | Sampling and decoding, speculative decoding | Holtzman 2020; Leviathan 2023 |
| 10 | Quantization | Dettmers 2022; Frantar 2022 (GPTQ); Lin 2023 (AWQ) |
| 11 | LoRA fine-tuning | Hu 2021; Dettmers 2023 (QLoRA) |
| 12 | Alignment: RLHF and DPO | Ouyang 2022; Rafailov 2023 |
| 13 | Reasoning and RL | Wei 2022 (CoT); DeepSeek-R1 2025 |
| 14 | Capstone: train the GPT | nanoGPT |

Possible extras: Mixture of Experts, interpretability (induction heads, the IOI circuit).

## Hardware

Everything is designed for a laptop without a GPU (tested on an Intel i5, 15 GB RAM). Models are 1-10M parameters when trained from scratch, and pretrained models up to roughly 0.5B parameters are used for inference, LoRA and quantization. Ideas that only show their benefit on GPUs (FlashAttention, MoE throughput, speculative decoding speed) are implemented for correctness, and the speedups are explained rather than measured.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

The first notebook run downloads GPT-2 weights (about 500 MB) from Hugging Face.

## Running the notebooks

- **Locally (default):** `.venv/bin/jupyter lab`, then open a notebook from `notebooks/`.
- **Google Colab:** each module in the table above has an "Open in Colab" link (opens straight from this repo; Colab already ships with torch and transformers).
- **GitHub Codespaces:** Code > Codespaces > Create codespace on `main`. The dev container installs the dependencies, then open a notebook in the editor. Check your free monthly allowance first.
