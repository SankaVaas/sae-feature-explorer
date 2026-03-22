"""
sae/activations.py

Hooks into GPT-2 Small via TransformerLens and collects MLP output
activations for a given layer. These activations are the raw input
your SAE will be trained on.

Usage:
    from sae.activations import ActivationCollector

    collector = ActivationCollector(model_name="gpt2", layer=3)
    acts = collector.collect(prompts, hook_point="hook_mlp_out")
    collector.save(acts, "data/cached_activations/layer3_mlp_out.pt")
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import transformer_lens
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "gpt2"
DEFAULT_LAYER   = 3
DEFAULT_HOOK    = "hook_mlp_out"
DEFAULT_BATCH   = 32
DEFAULT_DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Wraps a HookedTransformer and extracts MLP activations at a chosen layer.

    Args:
        model_name:  TransformerLens model identifier, e.g. "gpt2".
        layer:       Transformer layer index (0-indexed). Layer 3 is richest
                     for factual recall in GPT-2 Small (empirically).
        device:      "cuda", "cpu", or "mps".
        center_acts: Subtract the mean activation vector before returning.
                     Helps SAE training converge faster.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        layer: int = DEFAULT_LAYER,
        device: str = DEFAULT_DEVICE,
        center_acts: bool = True,
    ):
        self.model_name  = model_name
        self.layer       = layer
        self.device      = device
        self.center_acts = center_acts

        print(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=True,   # standard for interp work
            center_unembed=True,
            fold_ln=True,
            device=device,
        )
        self.model.eval()
        self.d_model = self.model.cfg.d_model   # 768 for GPT-2 Small

        # Will be computed after first collection pass
        self._mean_activation: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def collect(
        self,
        prompts: List[str],
        hook_point: str = DEFAULT_HOOK,
        batch_size: int = DEFAULT_BATCH,
        seq_pos: str = "last",
    ) -> torch.Tensor:
        """
        Run prompts through the model and collect activations at the hook point.

        Args:
            prompts:    List of text strings to run through the model.
            hook_point: TransformerLens hook name within the chosen layer.
                        "hook_mlp_out"  — output of the MLP block (recommended)
                        "hook_resid_post" — full residual stream post-layer
            batch_size: Number of prompts per forward pass.
            seq_pos:    Which token position to collect.
                        "last"  — final token (default, captures the prediction step)
                        "all"   — every token position (returns shape [N*T, d_model])
                        int     — specific position index

        Returns:
            Tensor of shape [N, d_model] (seq_pos="last") or [N*T, d_model] (seq_pos="all")
            where N = number of prompts, T = sequence length.
        """
        full_hook = f"blocks.{self.layer}.{hook_point}"
        all_acts  = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
            batch = prompts[i : i + batch_size]

            tokens = self.model.to_tokens(batch, prepend_bos=True)  # [B, T]

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=full_hook,
                    device=self.device,
                )

            acts = cache[full_hook]   # [B, T, d_model]

            if seq_pos == "last":
                # Use the last *non-padding* token for each prompt.
                # For left-padded batches, that's always position -1.
                acts = acts[:, -1, :]          # [B, d_model]
            elif seq_pos == "all":
                B, T, D = acts.shape
                acts = acts.reshape(B * T, D)  # [B*T, d_model]
            elif isinstance(seq_pos, int):
                acts = acts[:, seq_pos, :]     # [B, d_model]
            else:
                raise ValueError(f"seq_pos must be 'last', 'all', or an int. Got: {seq_pos}")

            all_acts.append(acts.cpu())

        activations = torch.cat(all_acts, dim=0)   # [N, d_model]

        if self.center_acts:
            activations = self._center(activations)

        return activations

    def collect_from_dataset(
        self,
        dataset_path: str = "data/country_capitals.json",
        hook_point: str = DEFAULT_HOOK,
        n_variants: int = 5,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Convenience wrapper: loads the country-capitals JSON, generates prompt
        variants, and returns activations alongside metadata for each prompt.

        Args:
            dataset_path: Path to country_capitals.json
            hook_point:   Hook name (see collect()).
            n_variants:   Number of prompt phrasings per country pair.

        Returns:
            (activations [N, d_model], metadata list of dicts)
        """
        import json
        data = json.load(open(dataset_path))
        pairs = data["pairs"]

        prompts, metadata = [], []
        templates = _get_prompt_templates()[:n_variants]

        for pair in pairs:
            for tmpl in templates:
                prompt = tmpl.format(country=pair["country"])
                prompts.append(prompt)
                metadata.append({
                    "prompt":   prompt,
                    "country":  pair["country"],
                    "capital":  pair["capital"],
                    "template": tmpl,
                })

        acts = self.collect(prompts, hook_point=hook_point)
        return acts, metadata

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def save(
        self,
        activations: torch.Tensor,
        path: str,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """
        Save activations (and optional metadata) to disk.

        Saves a dict with keys:
            "activations" — float32 tensor [N, d_model]
            "metadata"    — list of dicts (if provided)
            "config"      — model/layer/hook info for reproducibility
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "activations": activations.float(),
            "metadata":    metadata or [],
            "config": {
                "model_name":  self.model_name,
                "layer":       self.layer,
                "d_model":     self.d_model,
                "centered":    self.center_acts,
                "mean_act":    self._mean_activation,
            },
        }
        torch.save(payload, path)
        mb = activations.numel() * 4 / 1e6
        print(f"Saved {activations.shape[0]} activations ({mb:.1f} MB) → {path}")

    @staticmethod
    def load(path: str) -> Tuple[torch.Tensor, List[dict], dict]:
        """
        Load activations saved by .save().

        Returns:
            (activations, metadata, config)
        """
        payload = torch.load(path, map_location="cpu")
        return payload["activations"], payload["metadata"], payload["config"]

    # -----------------------------------------------------------------------
    # Utility: probe the model's raw prediction for a prompt
    # -----------------------------------------------------------------------

    def get_top_predictions(
        self,
        prompt: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k next-token predictions for a prompt.
        Useful for sanity-checking that the model knows country-capitals.

        Example:
            collector.get_top_predictions("The capital of France is")
            → [(" Paris", 0.41), (" Paris", ...), ...]
        """
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = self.model(tokens)           # [1, T, vocab]
        last_logits = logits[0, -1, :]            # [vocab]
        probs       = torch.softmax(last_logits, dim=-1)
        top_probs, top_ids = probs.topk(k)

        return [
            (self.model.to_string(idx.item()), round(prob.item(), 4))
            for idx, prob in zip(top_ids, top_probs)
        ]

    def measure_task_accuracy(
        self,
        dataset_path: str = "data/country_capitals.json",
        template: str = "The capital of {country} is",
    ) -> float:
        """
        Baseline accuracy: what % of country-capital pairs does GPT-2 get right
        (i.e. the correct capital is the top-1 predicted token)?

        Returns:
            Accuracy as a float between 0 and 1.
        """
        import json
        pairs = json.load(open(dataset_path))["pairs"]

        correct = 0
        for pair in tqdm(pairs, desc="Measuring baseline accuracy"):
            prompt   = template.format(country=pair["country"])
            top1, _  = self.get_top_predictions(prompt, k=1)[0]
            # GPT-2 predicts " Paris" (with leading space) — strip for comparison
            if top1.strip().lower() == pair["capital"].strip().lower():
                correct += 1

        acc = correct / len(pairs)
        print(f"Baseline accuracy: {correct}/{len(pairs)} = {acc:.1%}")
        return acc

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _center(self, acts: torch.Tensor) -> torch.Tensor:
        """Subtract the mean activation vector (computed on this batch)."""
        if self._mean_activation is None:
            self._mean_activation = acts.mean(dim=0, keepdim=True)  # [1, d_model]
        return acts - self._mean_activation


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _get_prompt_templates() -> List[str]:
    """
    Multiple phrasings of the country-capital task.
    Using several variants makes the SAE features more robust —
    they have to generalise across surface form, not just memorise one template.
    """
    return [
        "The capital of {country} is",
        "What is the capital of {country}? The capital is",
        "{country}'s capital city is",
        "The capital city of {country} is",
        "Q: What is the capital of {country}? A:",
    ]


# ---------------------------------------------------------------------------
# Quick test — run this file directly to sanity-check your setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Activation collector smoke test ===\n")

    collector = ActivationCollector(model_name="gpt2", layer=3)

    # 1. Check that GPT-2 actually knows some capitals
    print("Top-5 predictions for 'The capital of France is':")
    for token, prob in collector.get_top_predictions("The capital of France is"):
        print(f"  {repr(token):15s}  {prob:.4f}")

    # 2. Collect activations for a few prompts
    sample_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
    ]
    acts = collector.collect(sample_prompts)
    print(f"\nActivation shape: {acts.shape}")   # expect [3, 768]
    print(f"Mean norm:        {acts.norm(dim=-1).mean():.2f}")

    # 3. Save and reload
    collector.save(acts, "/tmp/test_acts.pt")
    acts2, meta, cfg = ActivationCollector.load("/tmp/test_acts.pt")
    print(f"\nReloaded shape: {acts2.shape}")
    print(f"Config: {cfg}")
    print("\nSmoke test passed.")