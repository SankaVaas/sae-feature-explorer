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