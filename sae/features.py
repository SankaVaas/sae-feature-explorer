"""
sae/features.py

Feature analysis module — takes a trained SAE and answers:
  "What does each feature represent?"

Core workflow:
  1. Run a set of labelled prompts through GPT-2 + SAE
  2. For each feature, find which prompts activate it most strongly
  3. Look at those prompts and label the feature manually
  4. Build a feature → label map
  5. Identify which features are "country-capital" features

Usage:
    from sae.features import FeatureAnalyser

    analyser = FeatureAnalyser(sae, collector)

    # Find top activating prompts for every feature
    profile = analyser.profile_all_features(prompts, metadata)

    # Get the top features for a specific prompt
    analyser.explain_prompt("The capital of France is")

    # Find features that distinguish capital prompts from general text
    capital_features = analyser.find_task_features(
        task_prompts, general_prompts, top_k=20
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from sae.model import SparseAutoencoder
from sae.activations import ActivationCollector


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureProfile:
    """
    Everything we know about a single SAE feature after analysis.
    """
    feature_idx:    int
    top_prompts:    List[Tuple[str, float]]   # (prompt, activation_value), descending
    mean_activation: float
    max_activation:  float
    activation_freq: float                    # fraction of prompts that activate it
    label:           str = ""                 # human-assigned label
    is_task_feature: bool = False             # does it fire on capital prompts?

    def __repr__(self):
        label_str = f'"{self.label}"' if self.label else "unlabelled"
        return (
            f"Feature {self.feature_idx:4d} | {label_str:30s} | "
            f"freq={self.activation_freq:.2%}  max={self.max_activation:.3f}"
        )


@dataclass
class PromptAnalysis:
    """
    Feature breakdown for a single prompt.
    """
    prompt:         str
    top_features:   List[Tuple[int, float]]   # (feature_idx, activation), descending
    reconstruction: torch.Tensor              # [d_model] reconstructed activation
    original:       torch.Tensor              # [d_model] original activation
    recon_error:    float                     # L2 distance between original & recon


# ---------------------------------------------------------------------------
# FeatureAnalyser
# ---------------------------------------------------------------------------

class FeatureAnalyser:
    """
    Analyses a trained SAE to identify and label interpretable features.

    Args:
        sae:       Trained SparseAutoencoder
        collector: ActivationCollector (loaded model + hooks)
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
        self.device    = device
        self.d_model   = sae.cfg.d_model
        self.dict_size = sae.cfg.dict_size

        # Populated by profile_all_features()
        self.profiles:      Dict[int, FeatureProfile] = {}
        self.feature_labels: Dict[int, str]           = {}

    # -----------------------------------------------------------------------
    # Core: profile every feature
    # -----------------------------------------------------------------------

    def profile_all_features(
        self,
        prompts:   List[str],
        metadata:  Optional[List[dict]] = None,
        top_n:     int = 5,
        hook_point: str = "hook_mlp_out",
    ) -> Dict[int, FeatureProfile]:
        """
        For every feature in the SAE, find the prompts that activate it most.

        Args:
            prompts:   List of text prompts to analyse
            metadata:  Optional list of dicts with extra info per prompt
            top_n:     How many top-activating prompts to store per feature
            hook_point: Which hook to collect from

        Returns:
            Dict mapping feature_idx → FeatureProfile
        """
        print(f"Collecting activations for {len(prompts)} prompts...")
        acts = self.collector.collect(
            prompts,
            hook_point=hook_point,
            seq_pos="last",
        ).to(self.device)                          # [N, d_model]

        print("Encoding through SAE...")
        with torch.no_grad():
            features = self.sae.encode(acts)       # [N, dict_size]

        features_np = features.cpu().numpy()       # [N, dict_size]

        print("Building feature profiles...")
        profiles = {}

        for feat_idx in tqdm(range(self.dict_size), desc="Profiling features"):
            col    = features_np[:, feat_idx]      # [N] — activation for this feature
            active = col > 0

            if active.sum() == 0:
                # Dead feature
                profiles[feat_idx] = FeatureProfile(
                    feature_idx     = feat_idx,
                    top_prompts     = [],
                    mean_activation = 0.0,
                    max_activation  = 0.0,
                    activation_freq = 0.0,
                    label           = "[dead]",
                )
                continue

            # Top-N activating prompts
            top_idx  = np.argsort(col)[::-1][:top_n]
            top_prompts = [
                (prompts[i], float(col[i]))
                for i in top_idx
                if col[i] > 0
            ]

            profiles[feat_idx] = FeatureProfile(
                feature_idx     = feat_idx,
                top_prompts     = top_prompts,
                mean_activation = float(col[active].mean()),
                max_activation  = float(col.max()),
                activation_freq = float(active.mean()),
                label           = self.feature_labels.get(feat_idx, ""),
            )

        self.profiles = profiles
        alive = sum(1 for p in profiles.values() if p.label != "[dead]")
        print(f"\nProfiling complete: {alive}/{self.dict_size} features alive.")
        return profiles

    # -----------------------------------------------------------------------
    # Explain a single prompt
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def explain_prompt(
        self,
        prompt:    str,
        top_k:     int = 10,
        hook_point: str = "hook_mlp_out",
    ) -> PromptAnalysis:
        """
        Show which features activate for a given prompt and how strongly.

        Args:
            prompt:  Text string, e.g. "The capital of France is"
            top_k:   Number of top features to show
            hook_point: Hook name

        Returns:
            PromptAnalysis with ranked features
        """
        acts = self.collector.collect(
            [prompt], hook_point=hook_point, seq_pos="last"
        ).to(self.device)                          # [1, d_model]

        out  = self.sae(acts)
        feat = out.features[0]                     # [dict_size]

        # Top-k active features
        vals, idx = feat.topk(top_k)
        top_features = [
            (i.item(), v.item())
            for i, v in zip(idx, vals)
            if v > 0
        ]

        recon_error = F.mse_loss(out.reconstruction[0], acts[0]).item()

        # Print readable summary
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Reconstruction error: {recon_error:.5f}")
        print(f"\nTop {top_k} active features:")
        print(f"  {'Idx':>5}  {'Activation':>10}  {'Label'}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*30}")
        for feat_idx, val in top_features:
            label = self.feature_labels.get(feat_idx, "")
            bar   = "█" * int(val * 10 / max(v for _, v in top_features) + 0.5)
            print(f"  {feat_idx:>5}  {val:>10.4f}  {label or '(unlabelled)':30s}  {bar}")

        return PromptAnalysis(
            prompt         = prompt,
            top_features   = top_features,
            reconstruction = out.reconstruction[0].cpu(),
            original       = acts[0].cpu(),
            recon_error    = recon_error,
        )

    # -----------------------------------------------------------------------
    # Find task-specific features
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def find_task_features(
        self,
        task_prompts:    List[str],
        control_prompts: List[str],
        top_k:           int = 20,
        hook_point:       str = "hook_mlp_out",
    ) -> List[Tuple[int, float]]:
        """
        Find features that activate significantly MORE on task prompts
        (country-capital) than on control prompts (general text).

        This is the key analysis step — it identifies which features
        are causally relevant to the capital recall task.

        Args:
            task_prompts:    Prompts like "The capital of France is"
            control_prompts: General text prompts for comparison
            top_k:           How many discriminative features to return

        Returns:
            List of (feature_idx, discrimination_score) sorted descending.
            discrimination_score = mean_task_activation - mean_control_activation
        """
        print("Collecting task prompt activations...")
        task_acts = self.collector.collect(
            task_prompts, hook_point=hook_point, seq_pos="last"
        ).to(self.device)

        print("Collecting control prompt activations...")
        ctrl_acts = self.collector.collect(
            control_prompts, hook_point=hook_point, seq_pos="last"
        ).to(self.device)

        task_feats = self.sae.encode(task_acts)    # [N_task, dict_size]
        ctrl_feats = self.sae.encode(ctrl_acts)    # [N_ctrl, dict_size]

        task_mean  = task_feats.mean(dim=0)        # [dict_size]
        ctrl_mean  = ctrl_feats.mean(dim=0)        # [dict_size]

        # Discrimination score: how much more active on task vs control
        disc_score = (task_mean - ctrl_mean).cpu().numpy()

        top_idx    = np.argsort(disc_score)[::-1][:top_k]
        results    = [(int(i), float(disc_score[i])) for i in top_idx]

        # Mark these as task features in profiles
        for feat_idx, score in results:
            if feat_idx in self.profiles:
                self.profiles[feat_idx].is_task_feature = True

        print(f"\nTop {top_k} task-discriminating features:")
        print(f"  {'Idx':>5}  {'Score':>8}  {'Label'}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*30}")
        for feat_idx, score in results[:top_k]:
            label = self.feature_labels.get(feat_idx, "(unlabelled)")
            print(f"  {feat_idx:>5}  {score:>8.4f}  {label}")

        return results

    # -----------------------------------------------------------------------
    # Country-specific feature analysis
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def compare_countries(
        self,
        countries: List[str],
        template:  str = "The capital of {country} is",
        top_k:     int = 5,
        hook_point: str = "hook_mlp_out",
    ) -> pd.DataFrame:
        """
        For each country, get the top-k active features and build a
        comparison matrix.

        Shows whether different countries share features (e.g. a general
        "European country" feature) or have unique ones ("France-specific").

        Returns:
            DataFrame — rows=countries, columns=feature indices,
            values=activation strength.
        """
        records = {}

        for country in tqdm(countries, desc="Analysing countries"):
            prompt = template.format(country=country)
            acts   = self.collector.collect(
                [prompt], hook_point=hook_point, seq_pos="last"
            ).to(self.device)

            feats = self.sae.encode(acts)[0].cpu().numpy()   # [dict_size]
            records[country] = feats

        df = pd.DataFrame(records).T                        # [n_countries, dict_size]
        df.columns = [f"f{i}" for i in range(self.dict_size)]

        # Keep only features that activate for at least one country
        active_cols = df.columns[(df > 0).any()]
        df = df[active_cols]

        return df

    # -----------------------------------------------------------------------
    # Feature labelling
    # -----------------------------------------------------------------------

    def label_feature(self, feature_idx: int, label: str) -> None:
        """
        Manually assign a human-readable label to a feature.
        Labels are stored in self.feature_labels and synced to profiles.

        Usage:
            analyser.label_feature(47, "European country capital")
            analyser.label_feature(112, "geographic location")
        """
        self.feature_labels[feature_idx] = label
        if feature_idx in self.profiles:
            self.profiles[feature_idx].label = label
        print(f"Feature {feature_idx} labelled: '{label}'")

    def label_features_bulk(self, labels: Dict[int, str]) -> None:
        """Assign multiple labels at once from a dict."""
        for feat_idx, label in labels.items():
            self.label_feature(feat_idx, label)

    def save_labels(self, path: str) -> None:
        """Save feature labels to a JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.feature_labels, f, indent=2)
        print(f"Labels saved → {path}")

    def load_labels(self, path: str) -> None:
        """Load feature labels from a JSON file."""
        import json
        with open(path) as f:
            raw = json.load(f)
        # JSON keys are always strings — convert back to int
        self.feature_labels = {int(k): v for k, v in raw.items()}
        # Sync to profiles if they exist
        for feat_idx, label in self.feature_labels.items():
            if feat_idx in self.profiles:
                self.profiles[feat_idx].label = label
        print(f"Loaded {len(self.feature_labels)} labels from {path}")

    # -----------------------------------------------------------------------
    # Summary helpers
    # -----------------------------------------------------------------------

    def top_features_summary(self, n: int = 20) -> None:
        """Print the n most active features across all profiled prompts."""
        if not self.profiles:
            print("Run profile_all_features() first.")
            return

        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.max_activation,
            reverse=True,
        )[:n]

        print(f"\nTop {n} features by max activation:")
        print(f"  {'Idx':>5}  {'MaxAct':>7}  {'Freq':>6}  {'Label'}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*35}")
        for p in sorted_profiles:
            print(
                f"  {p.feature_idx:>5}  "
                f"{p.max_activation:>7.3f}  "
                f"{p.activation_freq:>6.2%}  "
                f"{p.label or '(unlabelled)'}"
            )

    def task_features_summary(self) -> None:
        """Print all features marked as task-relevant."""
        task = [p for p in self.profiles.values() if p.is_task_feature]
        if not task:
            print("No task features found. Run find_task_features() first.")
            return

        print(f"\n{len(task)} task-relevant features:")
        for p in sorted(task, key=lambda x: x.max_activation, reverse=True):
            print(f"  {p}")
            if p.top_prompts:
                for prompt, val in p.top_prompts[:2]:
                    print(f"      [{val:.3f}] {repr(prompt)[:60]}")

    def to_dataframe(self) -> pd.DataFrame:
        """Export all feature profiles to a pandas DataFrame."""
        rows = []
        for p in self.profiles.values():
            top1 = p.top_prompts[0][0] if p.top_prompts else ""
            rows.append({
                "feature_idx":    p.feature_idx,
                "label":          p.label,
                "max_activation": p.max_activation,
                "mean_activation":p.mean_activation,
                "activation_freq":p.activation_freq,
                "is_task_feature":p.is_task_feature,
                "top_prompt":     top1,
            })
        return pd.DataFrame(rows).set_index("feature_idx")


# ---------------------------------------------------------------------------
# Standalone helper: quick feature inspection without full analyser
# ---------------------------------------------------------------------------

@torch.no_grad()
def top_activating_features(
    sae:        SparseAutoencoder,
    activation: torch.Tensor,
    top_k:      int = 10,
) -> List[Tuple[int, float]]:
    """
    Quick one-liner: given a single activation vector, return top-k features.

    Args:
        sae:        Trained SAE
        activation: [d_model] tensor
        top_k:      Number of features to return

    Returns:
        List of (feature_idx, value) sorted descending.
    """
    if activation.dim() == 1:
        activation = activation.unsqueeze(0)
    features = sae.encode(activation.to(next(sae.parameters()).device))[0]
    vals, idx = features.topk(top_k)
    return [(i.item(), v.item()) for i, v in zip(idx, vals) if v > 0]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, os

    print("=== Feature analyser smoke test ===\n")

    # Load SAE
    from sae.model import SparseAutoencoder
    sae_path = "results/checkpoints/sae_final.pt"
    if not os.path.exists(sae_path):
        print(f"No trained SAE found at {sae_path}.")
        print("Train first: python -m sae.train --config configs/sae_small.yaml")
        exit(1)

    sae       = SparseAutoencoder.load(sae_path)
    collector = ActivationCollector(layer=3)
    analyser  = FeatureAnalyser(sae, collector)

    # 1. Explain a single prompt
    analysis = analyser.explain_prompt("The capital of France is")

    # 2. Quick task vs control comparison
    task_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Australia is",
    ]
    control_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Photosynthesis converts sunlight into energy.",
        "Water boils at one hundred degrees Celsius.",
        "Shakespeare wrote thirty-seven plays.",
        "The Amazon River is the largest river.",
    ]

    task_features = analyser.find_task_features(task_prompts, control_prompts, top_k=10)

    # 3. Profile all features on a small set
    all_prompts = task_prompts + control_prompts
    profiles    = analyser.profile_all_features(all_prompts, top_n=3)

    analyser.top_features_summary(n=10)

    print("\nSmoke test passed.")