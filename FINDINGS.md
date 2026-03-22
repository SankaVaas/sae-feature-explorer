# Research findings log

## Hypothesis
GPT-2 Small encodes country-capital associations via a sparse set of
identifiable MLP features that can be located, labelled, and causally
validated through ablation.

## Experiment log

### [DATE] — baseline
- Model: gpt2 (117M)
- Hook point: hook_mlp_out, layer 3
- Task accuracy before ablation: TBD

## Open questions
- Which layer holds the richest country-capital features?
- Does dict_size 512 vs 4096 change feature interpretability?
- Can ablating identified features causally reduce accuracy?

## Key metrics to track
- L0 sparsity (avg features active per token)
- L2 reconstruction loss
- Task accuracy pre/post ablation
- % dead features


## 2024-03-22 — First successful SAE training

- Activations: 4,656 vectors, layer 3 hook_mlp_out, GPT-2 Small
- Config: dict_size=512, l1_coeff=0.5, lr=1e-4, steps=5000, batch=512
- Results:
  - L0: 33.5 / 512 (6.5% sparsity) ✓
  - L2: 0.0125 (good reconstruction) ✓
  - Dead features: 0% ✓
- Note: l1_coeff=0.5 is high — revisit when scaling to dict_size=4096

## FINAL RESULTS — SAE ablation on GPT-2 Medium, layer 14 resid stream

### Setup
- Model: GPT-2 Medium (345M parameters)
- Task: Country-capital factual recall, few-shot Style 2 prompt
- Hook: hook_resid_post, layer 14
- SAE: dict_size=512, L0=149/512, zero dead features
- Baseline accuracy: 86.7% (13/15 countries, top-5)

### Key finding 1 — Capital information lives in attention stream
Layer-by-layer patching (France → Germany):
- Layers 0–13: no effect, Germany still predicts "Berlin"
- Layer 14 resid_post: FIRST transfers capital → predicts "Paris"
- Layer 14+ resid_post: all transfer capital identity
- hook_mlp_out at ANY layer: never transfers capital
- Conclusion: country-capital associations written by attention
  heads at layers 14–23, not by MLP layers

### Key finding 2 — Single features are causally necessary
Ablating feature 304 alone: 86.7% → 0.0% (100% relative drop)
Ablating feature 201 alone: 86.7% → 0.0% (100% relative drop)
Ablating feature 473 alone: 86.7% → 0.0% (100% relative drop)
Ablating feature 197 alone: 86.7% → 0.0% (100% relative drop)
Ablating feature 294 alone: 86.7% → 0.0% (100% relative drop)

Post-ablation predictions: "First", "Great", "Good", "Making"
— completely incoherent, not wrong capitals but nonsense.
This suggests these features gate the entire capital-recall
pathway, not just individual country associations.

### Key finding 3 — L0=149 suggests over-dense SAE
With L0=149/512 features active per token, the SAE is not
as sparse as the MLP-trained version (L0=3.6). The residual
stream is a much higher-dimensional representation. Consider
training with higher l1_coeff or larger dict_size to improve
feature separability.

### Open questions
1. Are features 304/201/473 country-specific or query-structure specific?
2. Do these same features activate on other factual recall tasks
   (president of X, author of Y)?
3. Why does ablating one feature destroy ALL capitals rather than
   specific ones? Suggests gating rather than storage.