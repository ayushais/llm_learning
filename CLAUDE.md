# CLAUDE.md

This repo is a personal LLM course: HTML lessons + runnable notebooks + a capstone GPT. The README defines the course structure and module list; follow it. The learner knows Python, PyTorch, linear algebra and backprop, so skip basics and go deep on the ideas.

## Layout

- `docs/` HTML lessons (`index.html`, `NN_name.html`), served by GitHub Pages from `main` `/docs`
- `notebooks/` `NN_name.ipynb`, same number and name as its lesson
- `capstone/` final GPT
- `tools/moduleNN/` scripts that generate a lesson's figures from real model data (`data.py` extracts data, `figs.py` writes SVG into the HTML and is idempotent)
- `.devcontainer/` Codespaces/VS Code container setup (CPU torch + requirements.txt)
- `.venv/` local environment (gitignored); run Python as `.venv/bin/python`

## Content rules

- **Ground everything in research.** Each technique cites its primary paper (arXiv or official link). Do not state numbers, results or attributions from memory without being sure; verify with a search or by running the code. Do not copy paper text or figures; summarize and link.
- **Real over schematic.** Prefer figures and numbers produced by running real models (GPT-2 etc.). Label anything illustrative as "schematic, not measured data".
- **CPU only.** Everything must run on a laptop CPU (i5, 15 GB RAM, no GPU) in minutes. Use tiny models for training from scratch and small pretrained models for inference. Explain GPU-only gains instead of measuring them.
- **Notebooks are standalone and executed.** Run them top to bottom with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace` before committing so outputs are saved. Fix seeds. Keep downloads and checkpoints out of git (`data/`, `*.pt`, `*.safetensors` are ignored). Write notebooks via a generator script or nbformat, then execute.
- **Implement from scratch first,** then compare to the reference implementation (PyTorch / Hugging Face) to prove it matches.
- Hugging Face note: load with `attn_implementation="eager"` when attention weights are needed; the default SDPA path does not return them.

## HTML lesson conventions

- One self-contained file per lesson: CSS inline in a `<style>` block (copy it from `docs/01_big_picture.html`), no external scripts, fonts or images, so it works offline and on a tablet. Only paper links go outside.
- Light and dark mode via CSS variables and `prefers-color-scheme`; max width about 46rem; no horizontal scroll at 390px.
- Figures: inline SVG, `viewBox` width 480, text at least 11px, each in `<figure class="fig" id="figN">` with `<figcaption><b>Figure N.</b> ...</figcaption>`. Series colors `--s1`, `--s2`, `--s3` (the validated palette: blue, orange, aqua), fixed order. Label bars and lines directly; add `<title>` tooltips to marks. Math is plain HTML/Unicode (add inlined KaTeX only if a module truly needs it).
- Sections follow the README's lesson structure, end with a reading list and check-yourself questions, and link prev/next at the bottom. Update `docs/index.html` (move the module from "coming" to a link) and the README module table when a module lands.
- After editing HTML or figures, render with headless Chrome (`google-chrome --headless --no-sandbox --screenshot=...`) and look at the result; check for clipped labels and overlaps before reporting done.

## Definition of done for a module

1. Lesson HTML written with figures, citations, reading list and questions.
2. Notebook executed with outputs saved; exercises included.
3. Figure scripts committed under `tools/moduleNN/` if figures use model data.
4. `docs/index.html` and README table updated, including the module's Colab link (`https://colab.research.google.com/github/ayushais/llm_learning/blob/main/notebooks/NN_name.ipynb`) in both the README and the lesson's code-companion card.
5. Lesson screenshot checked at desktop and phone width.

## Git and publishing

- Remote: `https://github.com/ayushais/llm_learning` (public), HTTPS via the `gh` credential helper (SSH is not set up). Commit identity is configured locally (`ayush`).
- Commit and push only when the user asks. Pushing publishes to a public repo and GitHub Pages, so review the diff for anything that should not be public (tokens, personal data, large files) first.
- Show the user a new module before committing it.
