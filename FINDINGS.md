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