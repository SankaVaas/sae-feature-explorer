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