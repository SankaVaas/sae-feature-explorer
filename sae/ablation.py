"""
sae/ablation.py

Causal ablation experiments — the scientific proof that identified features
actually CAUSE the model's capital recall behaviour.

The core idea:
    1. Run GPT-2 normally → measure baseline capital recall accuracy
    2. Hook into the model, zero out specific SAE features during the
       forward pass, run again → measure accuracy after ablation
    3. If accuracy drops significantly → those features are causally
       responsible for the behaviour

This is what separates correlation from causation in mechanistic
interpretability. Finding that feature 468 activates on capital queries
is interesting. Proving that removing it breaks capital recall is a finding.

Methods implemented:
    - Single feature ablation
    - Feature cluster ablation (ablate a set together)
    - Activation patching (replace with mean activation)
    - Graduated ablation (ablate features one by one, track accuracy curve)

Usage:
    from sae.ablation import AblationExperiment

    exp = AblationExperiment(sae, collector)

    # Baseline accuracy
    baseline = exp.measure_accuracy(task_prompts, expected_capitals)

    # Ablate feature 468 and measure drop
    result = exp.ablate_features(
        prompts          = task_prompts,
        expected         = expected_capitals,
        feature_indices  = [468],
        method           = "zero",
    )

    # Full graduated ablation curve
    exp.graduated_ablation(task_prompts, expected_capitals, candidate_features)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae.model import SparseAutoencoder
from sae.activations import ActivationCollector


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    feature_indices:  List[int]
    method:           str            # "zero" or "mean"
    baseline_acc:     float
    ablated_acc:      float
    accuracy_drop:    float          # baseline - ablated
    relative_drop:    float          # accuracy_drop / baseline
    per_country:      Dict[str, Tuple[bool, bool]]  # country → (baseline_correct, ablated_correct)

    def __repr__(self):
        return (
            f"AblationResult(features={self.feature_indices}, "
            f"method={self.method}, "
            f"baseline={self.baseline_acc:.1%}, "
            f"ablated={self.ablated_acc:.1%}, "
            f"drop={self.accuracy_drop:.1%})"
        )

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"Ablation result — features {self.feature_indices}")
        print(f"{'='*55}")
        print(f"  Method:          {self.method}")
        print(f"  Baseline acc:    {self.baseline_acc:.1%}")
        print(f"  Ablated acc:     {self.ablated_acc:.1%}")
        print(f"  Accuracy drop:   {self.accuracy_drop:.1%}  "
              f"({self.relative_drop:.1%} relative)")
        print(f"\n  Per-country breakdown:")
        print(f"  {'Country':<15}  {'Baseline':>10}  {'Ablated':>10}  {'Changed?':>10}")
        print(f"  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*10}")
        for country, (base, abl) in sorted(self.per_country.items()):
            changed = "BROKE" if (base and not abl) else ("FIXED" if (not base and abl) else "same")
            b_str   = "correct" if base else "wrong"
            a_str   = "correct" if abl  else "wrong"
            marker  = " <--" if changed == "BROKE" else ""
            print(f"  {country:<15}  {b_str:>10}  {a_str:>10}  {changed:>10}{marker}")


@dataclass
class GraduatedAblationResult:
    """Results of ablating features one by one."""
    feature_order:    List[int]      # order features were ablated
    accuracies:       List[float]    # accuracy after ablating 0,1,2,...,N features
    baseline_acc:     float


# ---------------------------------------------------------------------------
# AblationExperiment
# ---------------------------------------------------------------------------

class AblationExperiment:
    """
    Runs causal ablation experiments on a trained SAE + GPT-2.

    The ablation works by:
        1. Collecting activations normally
        2. Encoding through SAE to get features
        3. Zeroing (or replacing) specific feature dimensions
        4. Decoding back to activation space
        5. Patching the modified activation back into the model
        6. Completing the forward pass and reading the output

    Args:
        sae:       Trained SparseAutoencoder
        collector: ActivationCollector with loaded GPT-2
        device:    "cuda" or "cpu"
    """

    def __init__(
        self,
        sae:       SparseAutoencoder,
        collector: ActivationCollector,
        device:    str = "cpu",
    ):
        self.sae       = sae.to(device).eval()
        self.collector = collector
        self.model     = collector.model
        self.device    = device
        self.layer     = collector.layer
        self.hook_pt   = f"blocks.{self.layer}.hook_mlp_out"

        # Cached mean activation — used for "mean ablation" method
        self._mean_feature_acts: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Baseline accuracy
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def measure_accuracy(
        self,
        prompts:   List[str],
        expected:  List[str],
        top_k:     int = 5,
        verbose:   bool = True,
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Measure how accurately GPT-2 predicts the correct capital.
        Checks if the expected capital appears in the top-k predictions.

        Args:
            prompts:  List of prompts e.g. ["The capital of France is", ...]
            expected: List of expected completions e.g. ["Paris", ...]
            top_k:    Consider correct if expected is in top-k predictions
            verbose:  Print per-prompt results

        Returns:
            (accuracy float, dict mapping prompt → correct bool)
        """
        assert len(prompts) == len(expected), "prompts and expected must match"

        results  = {}
        n_correct = 0

        for prompt, exp_capital in zip(prompts, expected):
            tokens   = self.model.to_tokens(prompt, prepend_bos=True)
            logits   = self.model(tokens)                # [1, T, vocab]
            last_log = logits[0, -1, :]                  # [vocab]
            top_ids  = last_log.topk(top_k).indices

            # Check if expected capital token appears in top-k
            # GPT-2 predicts " Paris" (with leading space)
            candidates = [
                exp_capital.strip(),
                " " + exp_capital.strip(),
                exp_capital.strip().lower(),
                " " + exp_capital.strip().lower(),
            ]
            predicted_tokens = [
                self.model.to_string(i.item()).strip().lower()
                for i in top_ids
            ]
            correct = any(
                c.strip().lower() in predicted_tokens
                for c in candidates
            )

            results[prompt] = correct
            if correct:
                n_correct += 1

            if verbose:
                top1    = self.model.to_string(last_log.argmax().item())
                status  = "✓" if correct else "✗"
                print(f"  {status} {prompt[:40]:<40} → {repr(top1):15s}  "
                      f"(expected: {exp_capital})")

        acc = n_correct / len(prompts)
        if verbose:
            print(f"\n  Accuracy: {n_correct}/{len(prompts)} = {acc:.1%}")

        return acc, results

    # -----------------------------------------------------------------------
    # Core ablation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ablate_features(
        self,
        prompts:         List[str],
        expected:        List[str],
        feature_indices: List[int],
        method:          str = "zero",
        verbose:         bool = True,
    ) -> AblationResult:
        """
        Ablate specific SAE features and measure accuracy change.

        How it works:
            For each prompt:
            1. Run model forward pass, capture MLP activation at self.layer
            2. Encode activation through SAE → feature vector
            3. Zero out (or replace with mean) the specified features
            4. Decode modified features back to activation space
            5. Patch this modified activation back into the residual stream
            6. Continue forward pass from that layer onward
            7. Read top-1 prediction

        Args:
            prompts:         List of prompts
            expected:        List of expected capital strings
            feature_indices: Which SAE feature dimensions to ablate
            method:          "zero"  → set features to 0
                             "mean"  → replace with mean activation value
            verbose:         Print per-prompt results

        Returns:
            AblationResult with full breakdown
        """
        assert method in ("zero", "mean"), "method must be 'zero' or 'mean'"

        # Compute mean feature activations if needed
        if method == "mean" and self._mean_feature_acts is None:
            self._compute_mean_features(prompts)

        if verbose:
            print(f"\nAblating features {feature_indices} (method={method})...")
            print(f"{'─'*60}")

        baseline_results = {}
        ablated_results  = {}

        for prompt, exp_capital in zip(prompts, expected):

            # --- Baseline prediction (no ablation) ---
            tokens   = self.model.to_tokens(prompt, prepend_bos=True)
            logits   = self.model(tokens)
            last_log = logits[0, -1, :]
            baseline_correct = self._is_correct(last_log, exp_capital)
            baseline_results[prompt] = baseline_correct

            # --- Ablated prediction ---
            ablated_logits = self._run_with_ablation(
                tokens, feature_indices, method
            )
            ablated_correct = self._is_correct(ablated_logits, exp_capital)
            ablated_results[prompt] = ablated_correct

            if verbose:
                b_top1 = self.model.to_string(last_log.argmax().item())
                a_top1 = self.model.to_string(ablated_logits.argmax().item())
                change = ""
                if baseline_correct and not ablated_correct:
                    change = " ← BROKE"
                elif not baseline_correct and ablated_correct:
                    change = " ← FIXED"
                print(f"  {prompt[:38]:<38}  "
                      f"base={repr(b_top1):10s}  "
                      f"abl={repr(a_top1):10s}{change}")

        # --- Compute summary stats ---
        baseline_acc = sum(baseline_results.values()) / len(prompts)
        ablated_acc  = sum(ablated_results.values())  / len(prompts)
        acc_drop     = baseline_acc - ablated_acc
        rel_drop     = acc_drop / baseline_acc if baseline_acc > 0 else 0

        # Build per-country dict
        per_country = {}
        for prompt, exp_capital in zip(prompts, expected):
            country = _extract_country(prompt)
            per_country[country] = (
                baseline_results[prompt],
                ablated_results[prompt],
            )

        result = AblationResult(
            feature_indices = feature_indices,
            method          = method,
            baseline_acc    = baseline_acc,
            ablated_acc     = ablated_acc,
            accuracy_drop   = acc_drop,
            relative_drop   = rel_drop,
            per_country     = per_country,
        )

        if verbose:
            result.print_summary()

        return result

    # -----------------------------------------------------------------------
    # Graduated ablation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def graduated_ablation(
        self,
        prompts:            List[str],
        expected:           List[str],
        candidate_features: List[int],
        method:             str = "zero",
    ) -> GraduatedAblationResult:
        """
        Ablate features one by one in order of discrimination score,
        plotting how accuracy degrades.

        This answers: "How many features do we need to ablate before
        capital recall breaks down completely?"

        Args:
            prompts:            Task prompts
            expected:           Expected capitals
            candidate_features: Feature indices ordered by importance
                                (e.g. from find_task_features())
            method:             "zero" or "mean"

        Returns:
            GraduatedAblationResult with accuracy at each step
        """
        print(f"\nGraduated ablation over {len(candidate_features)} features...")
        accuracies   = []
        ablated_so_far = []

        # Baseline (no ablation)
        baseline_acc, _ = self.measure_accuracy(prompts, expected, verbose=False)
        accuracies.append(baseline_acc)

        for feat_idx in tqdm(candidate_features, desc="Graduated ablation"):
            ablated_so_far.append(feat_idx)
            result = self.ablate_features(
                prompts, expected,
                feature_indices = ablated_so_far,
                method          = method,
                verbose         = False,
            )
            accuracies.append(result.ablated_acc)

        # Print curve
        print(f"\n  Features ablated  →  Accuracy")
        print(f"  {'─'*35}")
        print(f"  {'0 (baseline)':<20}  {baseline_acc:.1%}")
        for i, (feat, acc) in enumerate(zip(candidate_features, accuracies[1:])):
            drop = baseline_acc - acc
            bar  = "█" * int(drop * 30)
            print(f"  +F{feat:<17}  {acc:.1%}  drop={drop:.1%}  {bar}")

        return GraduatedAblationResult(
            feature_order = candidate_features,
            accuracies    = accuracies,
            baseline_acc  = baseline_acc,
        )

    # -----------------------------------------------------------------------
    # Activation patching
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def patch_with_country(
        self,
        source_country: str,
        target_country: str,
        template:       str = "The capital of {country} is",
    ) -> dict:
        """
        Activation patching experiment:
        Take the MLP activation from source_country's prompt and patch it
        into target_country's forward pass. Does the model now predict
        source_country's capital?

        This tests whether the entire activation (not just specific features)
        carries the country-capital association.

        Example:
            patch_with_country("France", "Germany")
            → If model now predicts "Paris" instead of "Berlin",
              the activation encodes "capital = Paris"

        Returns:
            dict with original prediction, patched prediction, and activations
        """
        src_prompt = template.format(country=source_country)
        tgt_prompt = template.format(country=target_country)

        src_tokens = self.model.to_tokens(src_prompt, prepend_bos=True)
        tgt_tokens = self.model.to_tokens(tgt_prompt, prepend_bos=True)

        # Get source activation
        _, src_cache = self.model.run_with_cache(
            src_tokens, names_filter=self.hook_pt
        )
        src_act = src_cache[self.hook_pt][0, -1, :]   # [d_model]

        # Get target baseline prediction
        tgt_logits = self.model(tgt_tokens)
        tgt_pred   = self.model.to_string(tgt_logits[0, -1, :].argmax().item())

        # Patch source activation into target forward pass
        def patch_hook(value, hook):
            value[0, -1, :] = src_act
            return value

        patched_logits = self.model.run_with_hooks(
            tgt_tokens,
            fwd_hooks=[(self.hook_pt, patch_hook)],
        )
        patched_pred = self.model.to_string(
            patched_logits[0, -1, :].argmax().item()
        )

        print(f"\nActivation patching: {source_country} → {target_country}")
        print(f"  Original prediction for '{target_country}': {repr(tgt_pred)}")
        print(f"  After patching '{source_country}' activation: {repr(patched_pred)}")

        return {
            "source_country":  source_country,
            "target_country":  target_country,
            "original_pred":   tgt_pred,
            "patched_pred":    patched_pred,
            "prediction_changed": tgt_pred != patched_pred,
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _run_with_ablation(
        self,
        tokens:          torch.Tensor,
        feature_indices: List[int],
        method:          str,
    ) -> torch.Tensor:
        """
        Run a forward pass with specified SAE features ablated.
        Returns logits for the last token position.
        """
        sae   = self.sae
        layer = self.layer

        def ablation_hook(value, hook):
            # value: [batch, seq, d_model]
            act = value[0, -1, :].unsqueeze(0)          # [1, d_model]

            # Encode → ablate → decode
            features = sae.encode(act)                  # [1, dict_size]

            if method == "zero":
                features[:, feature_indices] = 0.0
            elif method == "mean":
                mean_vals = self._mean_feature_acts[feature_indices]
                features[:, feature_indices] = mean_vals

            reconstruction = sae.decode(features)       # [1, d_model]

            # Replace the last token's activation with ablated version
            value[0, -1, :] = reconstruction[0]
            return value

        logits = self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_pt, ablation_hook)],
        )
        return logits[0, -1, :]                         # [vocab]

    def _is_correct(
        self,
        logits:      torch.Tensor,
        exp_capital: str,
        top_k:       int = 5,
    ) -> bool:
        """Check if expected capital is in top-k predictions."""
        top_ids = logits.topk(top_k).indices
        predicted = [
            self.model.to_string(i.item()).strip().lower()
            for i in top_ids
        ]
        candidates = [
            exp_capital.strip().lower(),
            exp_capital.strip().lower().split()[0],  # first word of capital
        ]
        return any(c in predicted for c in candidates)

    def _compute_mean_features(self, prompts: List[str]) -> None:
        """Compute mean feature activations across prompts for mean ablation."""
        acts = self.collector.collect(
            prompts, seq_pos="last"
        ).to(self.device)
        with torch.no_grad():
            features = self.sae.encode(acts)              # [N, dict_size]
        self._mean_feature_acts = features.mean(dim=0)    # [dict_size]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_country(prompt: str) -> str:
    """Extract country name from a capital query prompt."""
    # Handles "The capital of France is" → "France"
    lower = prompt.lower()
    if "capital of" in lower:
        parts = prompt.split("capital of")
        if len(parts) > 1:
            return parts[1].strip().split()[0].rstrip(" is")
    return prompt[:20]


def build_task_pairs(dataset_path: str = "data/country_capitals.json") -> Tuple[List[str], List[str]]:
    """
    Load country-capital pairs and return (prompts, expected_capitals).

    Returns:
        prompts:  ["The capital of France is", ...]
        expected: ["Paris", ...]
    """
    import json
    data   = json.load(open(dataset_path))
    pairs  = data["pairs"]
    tmpl   = data.get("prompt_template", "The capital of {country} is")

    prompts  = [tmpl.format(country=p["country"]) for p in pairs]
    expected = [p["capital"] for p in pairs]
    return prompts, expected


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, torch
    import torch.serialization
    from sae.model import SparseAutoencoder, SAEConfig

    print("=== Ablation smoke test ===\n")

    sae_path = "results/checkpoints/sae_final.pt"
    if not os.path.exists(sae_path):
        print(f"No SAE found at {sae_path}. Train first.")
        exit(1)

    torch.serialization.add_safe_globals([SAEConfig])
    payload = torch.load(sae_path, map_location="cpu", weights_only=True)
    sae     = SparseAutoencoder(payload["cfg"])
    sae.load_state_dict(payload["state_dict"])

    collector = ActivationCollector(layer=3)
    exp       = AblationExperiment(sae, collector)

    prompts, expected = build_task_pairs()

    # 1. Baseline
    print("Baseline accuracy:")
    baseline_acc, _ = exp.measure_accuracy(prompts, expected, top_k=5)

    # 2. Ablate feature 468 (most selective capital feature)
    result_468 = exp.ablate_features(
        prompts, expected,
        feature_indices=[468],
        method="zero",
    )

    # 3. Ablate the full candidate set
    candidates = [468, 396, 434, 410, 95, 11, 250]
    result_all = exp.ablate_features(
        prompts, expected,
        feature_indices=candidates,
        method="zero",
    )

    # 4. Activation patching
    exp.patch_with_country("France", "Germany")
    exp.patch_with_country("Japan",  "Brazil")

    print("\nSmoke test passed.")