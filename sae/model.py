"""
sae/model.py

Sparse Autoencoder (SAE) architecture for mechanistic interpretability.

The SAE learns to decompose dense MLP activations into a sparse set of
interpretable features. Core idea:
    activation (d_model,) → encode → features (dict_size,) [sparse]
                          → decode → reconstruction (d_model,)

Loss = reconstruction_loss (L2) + sparsity_penalty (L1 on features)

Reference: "Towards Monosemanticity" (Anthropic, 2023)
           "Toy Models of Superposition" (Elhage et al., 2022)

Usage:
    from sae.model import SparseAutoencoder, SAEConfig

    cfg = SAEConfig(d_model=768, dict_size=512, l1_coeff=1e-3)
    sae = SparseAutoencoder(cfg)

    # forward pass
    out = sae(activations)          # returns SAEOutput namedtuple
    print(out.loss, out.l2_loss, out.l1_loss)

    # get feature activations only
    features = sae.encode(activations)   # [batch, dict_size], sparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, NamedTuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    """
    All hyperparameters for the SAE in one place.
    Matches the fields in configs/sae_small.yaml and sae_large.yaml.
    """
    d_model:    int   = 768       # GPT-2 Small MLP output dim
    dict_size:  int   = 512       # number of learned features (> d_model for overcomplete basis)
    l1_coeff:   float = 1e-3      # sparsity penalty weight — higher = sparser features
    lr:         float = 1e-4      # Adam learning rate
    batch_size: int   = 4096      # activations per training step
    n_steps:    int   = 50_000    # total training steps

    # Dead feature revival — resample dead features periodically
    # A feature is "dead" if it never activates across a batch
    dead_feature_threshold: float = 1e-8
    dead_feature_revival_steps: int = 2000

    # Normalisation
    normalize_decoder: bool = True   # keep decoder columns unit norm (standard)

    seed: int = 42

    # Derived — set automatically, don't change
    expansion_factor: float = field(init=False)

    def __post_init__(self):
        self.expansion_factor = self.dict_size / self.d_model

    @classmethod
    def from_yaml(cls, path: str) -> "SAEConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        # Only pass fields that exist in the dataclass
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class SAEOutput(NamedTuple):
    """Everything returned by a forward pass — keeps the API clean."""
    reconstruction: torch.Tensor   # [batch, d_model]  — reconstructed activation
    features:       torch.Tensor   # [batch, dict_size] — sparse feature activations
    loss:           torch.Tensor   # scalar — total loss (l2 + l1)
    l2_loss:        torch.Tensor   # scalar — reconstruction loss
    l1_loss:        torch.Tensor   # scalar — sparsity loss
    l0:             torch.Tensor   # scalar — avg features active per token (diagnostic)


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    A one-hidden-layer autoencoder with ReLU activations and L1 sparsity.

    Architecture:
        encoder: Linear(d_model → dict_size) + bias + ReLU
        decoder: Linear(dict_size → d_model) + bias
                 (decoder columns kept unit-norm during training)

    The encoder learns a dictionary of directions in activation space.
    Each direction (feature) represents a human-interpretable concept,
    e.g. "this token is a European capital city".

    Key design decisions:
    - Separate encoder/decoder biases (not tied weights)
    - Decoder columns normalised to unit norm — prevents trivial solution
      of making one feature very large and ignoring the rest
    - L1 on post-ReLU features (not pre-ReLU) — penalises actual activations
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        torch.manual_seed(cfg.seed)

        # Encoder: d_model → dict_size
        self.encoder = nn.Linear(cfg.d_model, cfg.dict_size, bias=True)
        # Decoder: dict_size → d_model
        self.decoder = nn.Linear(cfg.dict_size, cfg.d_model, bias=True)

        # Initialise decoder columns to unit norm
        self._init_weights()

        # Track feature activation frequency for dead feature detection
        # Shape: [dict_size] — running mean of how often each feature fires
        self.register_buffer(
            "feature_activation_freq",
            torch.zeros(cfg.dict_size),
        )
        self.register_buffer(
            "steps_since_revival",
            torch.tensor(0),
        )

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """
        Args:
            x: [batch, d_model] — raw MLP activations (centered)

        Returns:
            SAEOutput namedtuple
        """
        # --- Encode ---
        features = self.encode(x)                    # [batch, dict_size], >= 0

        # --- Decode ---
        reconstruction = self.decode(features)       # [batch, d_model]

        # --- Loss ---
        l2_loss = self._reconstruction_loss(x, reconstruction)
        l1_loss = self._sparsity_loss(features)
        loss    = l2_loss + self.cfg.l1_coeff * l1_loss

        # --- Diagnostic: L0 (avg number of active features per token) ---
        l0 = (features > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            reconstruction=reconstruction,
            features=features,
            loss=loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            l0=l0,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse feature space.

        Args:
            x: [batch, d_model]

        Returns:
            features: [batch, dict_size], all values >= 0 (ReLU applied)
        """
        pre_relu = self.encoder(x)          # [batch, dict_size]
        features = F.relu(pre_relu)         # sparsity via ReLU
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct activations from sparse features.

        Args:
            features: [batch, dict_size]

        Returns:
            reconstruction: [batch, d_model]
        """
        return self.decoder(features)       # [batch, d_model]

    # -----------------------------------------------------------------------
    # Decoder normalisation (called after each optimiser step)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        """
        Project decoder columns back to unit norm.
        Call this after every optimiser step.

        Why: without this, the SAE can cheat by making decoder columns large
        and encoder weights small, achieving low loss without sparse features.
        Unit norm forces the model to actually use the feature directions.
        """
        if not self.cfg.normalize_decoder:
            return
        norms = self.decoder.weight.norm(dim=0, keepdim=True)   # [1, dict_size]
        self.decoder.weight.data /= norms.clamp(min=1e-8)

    # -----------------------------------------------------------------------
    # Dead feature tracking & revival
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def update_feature_stats(self, features: torch.Tensor) -> None:
        """
        Update the running estimate of how often each feature fires.
        Call this every training step with the current batch's features.
        Uses exponential moving average (alpha=0.99).
        """
        batch_freq = (features > 0).float().mean(dim=0)   # [dict_size]
        self.feature_activation_freq.mul_(0.99).add_(batch_freq * 0.01)

    def get_dead_features(self) -> torch.Tensor:
        """
        Return indices of features that have essentially never activated.
        These are wasted capacity — we revive them by re-initialising.

        Returns:
            LongTensor of dead feature indices
        """
        dead_mask = self.feature_activation_freq < self.cfg.dead_feature_threshold
        return dead_mask.nonzero(as_tuple=False).squeeze(-1)

    @torch.no_grad()
    def revive_dead_features(self, activations: torch.Tensor) -> int:
        """
        Reinitialise dead feature encoder rows to point toward high-loss
        activation directions, giving them a chance to become useful.

        This is the "neuron resampling" trick from Anthropic's SAE work.
        Without it, ~30% of features can die early and never recover.

        Args:
            activations: [batch, d_model] — a recent batch of activations

        Returns:
            Number of features revived.
        """
        dead_idx = self.get_dead_features()
        n_dead   = len(dead_idx)

        if n_dead == 0:
            return 0

        # Sample activation vectors proportional to reconstruction loss
        with torch.no_grad():
            out    = self.forward(activations)
            losses = F.mse_loss(
                out.reconstruction, activations, reduction="none"
            ).sum(dim=-1)                            # [batch]
            probs  = (losses / losses.sum()).cpu().numpy()

        import numpy as np
        chosen = np.random.choice(len(activations), size=n_dead, p=probs, replace=True)
        new_directions = activations[chosen]         # [n_dead, d_model]

        # Normalise and assign to encoder
        new_directions = F.normalize(new_directions, dim=-1)
        self.encoder.weight.data[dead_idx] = new_directions

        # Reset decoder columns for these features too
        self.decoder.weight.data[:, dead_idx] = new_directions.T

        # Reset their activation frequency
        self.feature_activation_freq[dead_idx] = 0.0

        print(f"  Revived {n_dead} dead features.")
        return n_dead

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights + config to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "cfg":        self.cfg,
        }, path)
        print(f"SAE saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        """Load a saved SAE checkpoint."""
        payload = torch.load(path, map_location=device)
        sae     = cls(payload["cfg"])
        sae.load_state_dict(payload["state_dict"])
        sae.eval()
        return sae

    # -----------------------------------------------------------------------
    # Feature inspection helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_active_features(
        self,
        x: torch.Tensor,
        top_k: int = 10,
    ) -> list:
        """
        For a single activation vector, return the top-k active features
        sorted by activation strength.

        Args:
            x:     [d_model] or [1, d_model]
            top_k: how many features to return

        Returns:
            List of (feature_idx, activation_value) tuples, descending.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.encode(x)[0]                    # [dict_size]
        vals, idx = features.topk(top_k)
        return [(i.item(), v.item()) for i, v in zip(idx, vals) if v > 0]

    @torch.no_grad()
    def get_decoder_direction(self, feature_idx: int) -> torch.Tensor:
        """
        Return the decoder direction for a given feature.
        This is the direction in activation space the feature "represents".

        Args:
            feature_idx: index into [dict_size]

        Returns:
            [d_model] unit vector
        """
        return self.decoder.weight[:, feature_idx].clone()   # [d_model]

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming uniform for encoder, unit-norm columns for decoder."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.normalise_decoder()    # start with unit-norm decoder columns

    @staticmethod
    def _reconstruction_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared L2 reconstruction loss, normalised by d_model."""
        return F.mse_loss(x_hat, x, reduction="mean")

    @staticmethod
    def _sparsity_loss(features: torch.Tensor) -> torch.Tensor:
        """L1 norm of feature activations — encourages sparsity."""
        return features.abs().mean()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SAE smoke test ===\n")

    cfg = SAEConfig(d_model=768, dict_size=512, l1_coeff=1e-3)
    sae = SparseAutoencoder(cfg)

    print(f"Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"Expansion factor: {cfg.expansion_factor:.1f}x  "
          f"({cfg.d_model} → {cfg.dict_size})\n")

    # Fake activations — replace with real ones from ActivationCollector
    x = torch.randn(64, cfg.d_model)

    out = sae(x)
    print(f"Input shape:          {x.shape}")
    print(f"Reconstruction shape: {out.reconstruction.shape}")
    print(f"Features shape:       {out.features.shape}")
    print(f"Total loss:           {out.loss.item():.4f}")
    print(f"  L2 (reconstruction):{out.l2_loss.item():.4f}")
    print(f"  L1 (sparsity):      {out.l1_loss.item():.4f}")
    print(f"  L0 (avg active):    {out.l0.item():.1f} / {cfg.dict_size}")

    # Check top active features for one vector
    top = sae.get_active_features(x[0], top_k=5)
    print(f"\nTop 5 active features for x[0]: {top}")

    # Check dead features (all dead at init since no training)
    dead = sae.get_dead_features()
    print(f"\nDead features at init: {len(dead)} / {cfg.dict_size}  (expected ~all)")

    # Save and reload
    sae.save("/tmp/test_sae.pt")
    sae2 = SparseAutoencoder.load("/tmp/test_sae.pt")
    out2 = sae2(x)
    assert torch.allclose(out.loss, out2.loss), "Save/load mismatch!"
    print("\nSave/load check passed.")
    print("\nSmoke test passed.")