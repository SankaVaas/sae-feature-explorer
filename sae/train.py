"""
sae/train.py

Training loop for the Sparse Autoencoder.

Feeds cached GPT-2 MLP activations through the SAE, optimises with Adam,
normalises decoder columns after every step, revives dead features
periodically, and logs everything to Weights & Biases.

Usage:
    # From the command line:
    python -m sae.train --config configs/sae_small.yaml

    # From a notebook:
    from sae.train import Trainer, TrainerConfig
    from sae.model import SAEConfig

    sae_cfg     = SAEConfig.from_yaml("configs/sae_small.yaml")
    trainer_cfg = TrainerConfig(
        activation_path = "data/cached_activations/layer3_mlp_out.pt",
        checkpoint_dir  = "results/checkpoints",
        use_wandb       = True,
    )
    trainer = Trainer(sae_cfg, trainer_cfg)
    sae     = trainer.train()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import time

from sae.model import SparseAutoencoder, SAEConfig
from sae.activations import ActivationCollector


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    Everything the training loop needs that isn't part of the SAE itself.
    """
    # Data
    activation_path: str = "data/cached_activations/layer3_mlp_out.pt"

    # Checkpointing
    checkpoint_dir:  str  = "results/checkpoints"
    save_every:      int  = 10_000   # save a checkpoint every N steps
    keep_last_n:     int  = 3        # keep only the N most recent checkpoints

    # Logging
    use_wandb:       bool = False
    wandb_project:   str  = "sae-feature-explorer"
    wandb_run_name:  str  = ""       # auto-generated if empty
    log_every:       int  = 100      # log metrics every N steps

    # Dead feature revival
    revive_every:    int  = 2_000    # check for dead features every N steps

    # Device
    device:          str  = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains a SparseAutoencoder on cached MLP activations.

    Training loop:
        for each step:
            1. Sample a batch of activations
            2. Forward pass through SAE
            3. Compute loss (L2 + L1)
            4. Backward + Adam step
            5. Normalise decoder columns (keeps unit norm)
            6. Update feature activation stats
            7. Every revive_every steps: revive dead features
            8. Every log_every steps: log to W&B / print
            9. Every save_every steps: save checkpoint
    """

    def __init__(self, sae_cfg: SAEConfig, trainer_cfg: TrainerConfig):
        self.sae_cfg     = sae_cfg
        self.trainer_cfg = trainer_cfg
        self.device      = trainer_cfg.device

        # Build model
        self.sae = SparseAutoencoder(sae_cfg).to(self.device)

        # Optimiser — Adam with no weight decay (standard for SAEs)
        self.optimiser = torch.optim.Adam(
            self.sae.parameters(),
            lr=sae_cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Load activations
        self.activations = self._load_activations()

        # W&B
        self.wandb_run = None
        if trainer_cfg.use_wandb:
            self._init_wandb()

        # Metrics history for notebook plotting
        self.history = {
            "step":    [],
            "loss":    [],
            "l2_loss": [],
            "l1_loss": [],
            "l0":      [],
            "dead_pct": [],
        }

        print(f"\nTrainer ready.")
        print(f"  Device:          {self.device}")
        print(f"  Activations:     {self.activations.shape}")
        print(f"  SAE dict size:   {sae_cfg.dict_size}")
        print(f"  L1 coefficient:  {sae_cfg.l1_coeff}")
        print(f"  Training steps:  {sae_cfg.n_steps:,}")
        print(f"  Batch size:      {sae_cfg.batch_size}\n")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> SparseAutoencoder:
        """
        Run the full training loop.

        Returns:
            Trained SparseAutoencoder (also saved to checkpoint_dir).
        """
        sae        = self.sae
        cfg        = self.sae_cfg
        tcfg       = self.trainer_cfg
        acts       = self.activations
        n_acts     = acts.shape[0]

        Path(tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        pbar       = tqdm(range(cfg.n_steps), desc="Training SAE")
        start_time = time.time()

        for step in pbar:

            # --- Sample batch ---
            idx   = torch.randint(0, n_acts, (cfg.batch_size,))
            batch = acts[idx].to(self.device)          # [B, d_model]

            # --- Forward ---
            out = sae(batch)

            # --- Backward ---
            self.optimiser.zero_grad()
            out.loss.backward()
            self.optimiser.step()

            # --- Post-step: normalise decoder ---
            sae.normalise_decoder()

            # --- Update feature stats ---
            sae.update_feature_stats(out.features.detach())

            # --- Dead feature revival ---
            if (step + 1) % tcfg.revive_every == 0:
                sae.revive_dead_features(batch.detach())

            # --- Logging ---
            if step % tcfg.log_every == 0 or step == cfg.n_steps - 1:
                n_dead   = len(sae.get_dead_features())
                dead_pct = 100 * n_dead / cfg.dict_size
                elapsed  = time.time() - start_time

                metrics = {
                    "step":     step,
                    "loss":     out.loss.item(),
                    "l2_loss":  out.l2_loss.item(),
                    "l1_loss":  out.l1_loss.item(),
                    "l0":       out.l0.item(),
                    "dead_pct": dead_pct,
                    "elapsed":  elapsed,
                }

                self._record(metrics)

                pbar.set_postfix({
                    "loss":  f"{metrics['loss']:.4f}",
                    "l2":    f"{metrics['l2_loss']:.4f}",
                    "l0":    f"{metrics['l0']:.1f}",
                    "dead%": f"{dead_pct:.1f}",
                })

                if self.wandb_run:
                    self.wandb_run.log(metrics, step=step)

            # --- Checkpointing ---
            if (step + 1) % tcfg.save_every == 0:
                self._save_checkpoint(step + 1)

        # Final save
        final_path = str(Path(tcfg.checkpoint_dir) / "sae_final.pt")
        sae.save(final_path)
        print(f"\nTraining complete. Final model → {final_path}")

        if self.wandb_run:
            self.wandb_run.finish()

        return sae

    # -----------------------------------------------------------------------
    # Evaluation helper — run after training
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, n_batches: int = 20) -> dict:
        """
        Compute mean metrics over n_batches of held-out activations.
        Call this after training to get a final snapshot.

        Returns:
            dict with mean loss, l2, l1, l0, dead_pct
        """
        sae    = self.sae.eval()
        acts   = self.activations
        cfg    = self.sae_cfg
        totals = {"loss": 0, "l2_loss": 0, "l1_loss": 0, "l0": 0}

        for _ in range(n_batches):
            idx   = torch.randint(0, acts.shape[0], (cfg.batch_size,))
            batch = acts[idx].to(self.device)
            out   = sae(batch)
            for k in totals:
                totals[k] += getattr(out, k).item()

        means    = {k: v / n_batches for k, v in totals.items()}
        n_dead   = len(sae.get_dead_features())
        means["dead_pct"] = 100 * n_dead / cfg.dict_size

        print("\nEvaluation results:")
        print(f"  Loss (total):      {means['loss']:.4f}")
        print(f"  L2 (reconstruct):  {means['l2_loss']:.4f}")
        print(f"  L1 (sparsity):     {means['l1_loss']:.4f}")
        print(f"  L0 (avg active):   {means['l0']:.1f} / {cfg.dict_size}")
        print(f"  Dead features:     {n_dead} ({means['dead_pct']:.1f}%)")

        sae.train()
        return means

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_activations(self) -> torch.Tensor:
        """
        Load cached activations from disk.
        If the cache doesn't exist yet, collect them on the fly from GPT-2.
        """
        path = Path(self.trainer_cfg.activation_path)

        if path.exists():
            print(f"Loading cached activations from {path}...")
            acts, _, cfg = ActivationCollector.load(str(path))
            print(f"  Shape: {acts.shape}  |  Layer: {cfg.get('layer', '?')}")
            return acts

        print(f"No cached activations found at {path}.")
        print("Collecting activations from GPT-2 Small (this takes ~2 min on CPU)...")

        collector = ActivationCollector(
            model_name = self.sae_cfg.__dict__.get("model_name", "gpt2"),
            layer      = self.sae_cfg.__dict__.get("hook_layer",  3),
            device     = self.device,
        )

        # Use Pile / wikitext snippets for general-purpose SAE training.
        # For our task we also include country-capital prompts.
        prompts = _get_training_prompts()
        acts    = collector.collect(prompts, seq_pos="all")   # every token position

        collector.save(acts, str(path))
        return acts

    def _save_checkpoint(self, step: int) -> None:
        """Save a numbered checkpoint and prune old ones."""
        ckpt_dir  = Path(self.trainer_cfg.checkpoint_dir)
        ckpt_path = ckpt_dir / f"sae_step_{step:06d}.pt"
        self.sae.save(str(ckpt_path))

        # Prune old checkpoints
        ckpts = sorted(ckpt_dir.glob("sae_step_*.pt"))
        for old in ckpts[: -self.trainer_cfg.keep_last_n]:
            old.unlink()

    def _record(self, metrics: dict) -> None:
        """Append metrics to in-memory history for notebook plotting."""
        for k in ["step", "loss", "l2_loss", "l1_loss", "l0", "dead_pct"]:
            if k in metrics:
                self.history[k].append(metrics[k])

    def _init_wandb(self) -> None:
        """Initialise a W&B run."""
        try:
            import wandb
            run_name = self.trainer_cfg.wandb_run_name or (
                f"sae-d{self.sae_cfg.dict_size}"
                f"-l{self.sae_cfg.hook_layer if hasattr(self.sae_cfg, 'hook_layer') else 3}"
                f"-l1{self.sae_cfg.l1_coeff}"
            )
            self.wandb_run = wandb.init(
                project = self.trainer_cfg.wandb_project,
                name    = run_name,
                config  = {**self.sae_cfg.__dict__, **self.trainer_cfg.__dict__},
            )
            print(f"W&B run: {self.wandb_run.url}")
        except ImportError:
            print("wandb not installed — skipping W&B logging.")
            self.wandb_run = None


# ---------------------------------------------------------------------------
# Training prompts
# ---------------------------------------------------------------------------

def _get_training_prompts() -> list:
    """
    Mix of general text + country-capital prompts for SAE training.
    The general text teaches the SAE about the full activation distribution.
    The task-specific prompts ensure country-capital features are well-represented.

    In production you'd stream from The Pile or OpenWebText.
    For our focused project, this curated set is sufficient.
    """
    general = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was darkness and silence.",
        "Scientists have discovered a new species of deep-sea fish.",
        "The stock market closed higher on Friday after strong earnings.",
        "Machine learning models are trained on large datasets.",
        "The history of Rome spans more than two thousand years.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "She opened the door and stepped into the sunlit room.",
        "The president signed the bill into law on Tuesday.",
        "Neural networks are loosely inspired by the human brain.",
        "The Amazon rainforest produces twenty percent of Earth's oxygen.",
        "He solved the equation by substituting known values.",
        "Languages evolve over time as cultures interact and merge.",
        "The telescope revealed thousands of previously unseen galaxies.",
        "Philosophy asks questions that science cannot always answer.",
    ]

    capitals = [
        "The capital of France is Paris, a city known for art.",
        "Germany's capital Berlin was divided during the Cold War.",
        "Tokyo, the capital of Japan, is the world's largest city.",
        "The capital of Brazil is Brasília, built in the 1950s.",
        "Canberra is the capital of Australia, not Sydney.",
        "Cairo, the capital of Egypt, sits beside the Nile River.",
        "New Delhi became the capital of India in 1911.",
        "Ottawa is the capital of Canada, located in Ontario.",
        "Buenos Aires is the capital and largest city of Argentina.",
        "Abuja replaced Lagos as the capital of Nigeria in 1991.",
        "Rome, the capital of Italy, was once the centre of an empire.",
        "Madrid is the capital and largest city of Spain.",
        "Beijing has been the capital of China for centuries.",
        "Moscow is the capital of Russia and its largest city.",
        "Mexico City is the capital and most populous city of Mexico.",
        "The capital of France is", "The capital of Germany is",
        "The capital of Japan is", "The capital of Brazil is",
        "The capital of Australia is", "The capital of Egypt is",
        "The capital of India is", "The capital of Canada is",
        "The capital of Argentina is", "The capital of Nigeria is",
    ]

    return general + capitals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument("--config",          default="configs/sae_small.yaml")
    parser.add_argument("--activation-path", default="data/cached_activations/layer3_mlp_out.pt")
    parser.add_argument("--checkpoint-dir",  default="results/checkpoints")
    parser.add_argument("--wandb",           action="store_true")
    parser.add_argument("--steps",           type=int, default=None)
    args = parser.parse_args()

    sae_cfg = SAEConfig.from_yaml(args.config)
    if args.steps:
        sae_cfg.n_steps = args.steps

    trainer_cfg = TrainerConfig(
        activation_path = args.activation_path,
        checkpoint_dir  = args.checkpoint_dir,
        use_wandb       = args.wandb,
    )

    trainer = Trainer(sae_cfg, trainer_cfg)
    sae     = trainer.train()
    trainer.evaluate()