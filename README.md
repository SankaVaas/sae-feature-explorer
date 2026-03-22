# SAE feature explorer

Mechanistic interpretability via sparse autoencoders on GPT-2 Small.
We decompose MLP activations to find human-interpretable features
responsible for country-capital factual recall, then validate them causally
via targeted ablation.

## Research question
Which internal features does GPT-2 Small use to recall that
"The capital of France is → Paris"? Can we find, label, and surgically
remove them?

## Quickstart (Colab)
Open `notebooks/01_collect_activations.ipynb` — all dependencies install
in the first cell.

## Local setup
pip install -r requirements.txt
pip install -e .

## Structure
| Path | Purpose |
|---|---|
| `sae/` | Core SAE model, training, analysis |
| `data/` | Country-capital dataset & prompt templates |
| `notebooks/` | Step-by-step Colab notebooks |
| `explorer/` | Interactive Gradio feature browser |
| `configs/` | SAE hyperparameter configs |
| `FINDINGS.md` | Living research log |


## Key findings

**Finding 1 — Capital information lives in attention, not MLP**  
Layer-by-layer patching shows capital identity first appears in the 
residual stream at layer 14. The MLP stream at any layer carries no 
capital-specific information.

**Finding 2 — Single features causally gate recall**  
Ablating any one of features {304, 201, 473, 197, 294} drops accuracy 
from 86.7% to 0%. These features gate the entire recall pathway rather 
than storing individual country-capital pairs.

**Finding 3 — Recall difficulty correlates with country frequency**  
GPT-2 Medium fails on Brazil (Brasília) and Nigeria (Abuja) — both 
have historically obscure or recently-changed capitals — suggesting 
factual knowledge correlates with training data frequency.

## Demo
[Live Gradio explorer] ← add your gradio.live URL here

## Setup
pip install -r requirements.txt
python -m explorer.app

## Results
| Metric | Value |
|--------|-------|
| Model | GPT-2 Medium (345M) |
| Hook | hook_resid_post, layer 14 |
| SAE dict size | 512 |
| Baseline accuracy | 86.7% |
| Accuracy after ablating F304 | 0.0% |
| Dead features | 0% |