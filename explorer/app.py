"""
explorer/app.py

Interactive Gradio app for exploring SAE features on GPT-2 Medium.

Features:
    - Enter any prompt and see which SAE features activate
    - Click a feature to see its top activating prompts
    - Run ablation experiments interactively
    - Visualise feature activation heatmaps across countries
    - Activation patching between countries

Usage:
    python -m explorer.app
    # or from notebook:
    from explorer.app import build_app
    app = build_app()
    app.launch()
"""

import torch
import torch.serialization
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import gradio as gr

from sae.model import SparseAutoencoder, SAEConfig
from sae.activations import ActivationCollector
from sae.features import FeatureAnalyser
from sae.ablation import AblationExperiment, build_task_pairs


# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------

STATE = {
    "sae":       None,
    "collector": None,
    "analyser":  None,
    "exp":       None,
    "labels":    {},
    "profiles":  None,
    "device":    "cuda" if torch.cuda.is_available() else "cpu",
}

HOOK_POINT   = "hook_resid_post"
LAYER        = 14
MODEL_NAME   = "gpt2-medium"
CHECKPOINT   = "results/checkpoints_resid14/sae_final.pt"
LABELS_PATH  = "results/feature_labels_resid14.json"
DATASET_PATH = "data/country_capitals.json"


def load_models():
    """Load SAE + GPT-2 Medium once at startup."""
    device = STATE["device"]
    print(f"Loading models on {device}...")

    # Load SAE
    torch.serialization.add_safe_globals([SAEConfig])
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    sae     = SparseAutoencoder(payload["cfg"])
    sae.load_state_dict(payload["state_dict"])
    sae.eval()

    # Load collector — override hook point
    collector = ActivationCollector(
        model_name=MODEL_NAME, layer=LAYER, device=device
    )
    original_collect = collector.collect
    collector.collect = lambda p, **kw: original_collect(
        p,
        hook_point=HOOK_POINT,
        **{k: v for k, v in kw.items() if k != "hook_point"},
    )

    # Load analyser
    analyser = FeatureAnalyser(sae, collector, device=device)

    # Load ablation experiment
    exp         = AblationExperiment(sae, collector, device=device)
    exp.hook_pt = f"blocks.{LAYER}.{HOOK_POINT}"

    # Load labels
    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            labels = {int(k): v for k, v in json.load(f).items()}
        analyser.feature_labels = labels

    STATE.update({
        "sae":      sae,
        "collector": collector,
        "analyser":  analyser,
        "exp":       exp,
        "labels":    labels,
    })

    print(f"Models loaded. SAE dict_size={sae.cfg.dict_size}, d_model={sae.cfg.d_model}")
    return sae, collector, analyser, exp


# ---------------------------------------------------------------------------
# Tab 1 — Prompt analyser
# ---------------------------------------------------------------------------

def analyse_prompt(prompt: str, top_k: int = 10):
    """Analyse a single prompt and return feature activations."""
    if not prompt.strip():
        return "Enter a prompt above.", None, ""

    sae       = STATE["sae"]
    collector = STATE["collector"]
    labels    = STATE["labels"]
    device    = STATE["device"]

    # Collect activation
    acts  = collector.collect([prompt], seq_pos="last").to(device)
    out   = sae(acts)
    feats = out.features[0].cpu()                # [dict_size]

    # Top-k active features
    vals, idx = feats.topk(int(top_k))
    rows = []
    for i, v in zip(idx.tolist(), vals.tolist()):
        if v > 0:
            rows.append({
                "Feature":    i,
                "Activation": round(v, 4),
                "Label":      labels.get(i, "(unlabelled)"),
                "Bar":        "█" * min(int(v * 5), 30),
            })

    if not rows:
        return "No features activated for this prompt.", None, ""

    df = pd.DataFrame(rows)

    # Top prediction from model
    top5 = collector.model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = collector.model(top5)
    top_preds = []
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_p, top_i = probs.topk(5)
    for prob, idx_t in zip(top_p, top_i):
        tok = collector.model.to_string(idx_t.item())
        top_preds.append(f"{repr(tok)} ({prob.item():.3f})")

    recon_err = out.l2_loss.item()
    summary = (
        f"**Reconstruction error:** {recon_err:.5f}  \n"
        f"**Active features:** {(feats > 0).sum().item()} / {sae.cfg.dict_size}  \n"
        f"**Top-5 next token predictions:** {', '.join(top_preds)}"
    )

    return summary, df, ""


def get_feature_detail(feature_idx_str: str):
    """Show details for a specific feature index."""
    try:
        feature_idx = int(feature_idx_str)
    except ValueError:
        return "Enter a valid feature index (integer)."

    labels   = STATE["labels"]
    analyser = STATE["analyser"]
    profiles = STATE.get("profiles")

    label = labels.get(feature_idx, "(unlabelled)")
    out   = [f"**Feature {feature_idx}** — {label}\n"]

    if profiles and feature_idx in profiles:
        p = profiles[feature_idx]
        out.append(f"- Max activation: {p.max_activation:.4f}")
        out.append(f"- Activation freq: {p.activation_freq:.1%}")
        out.append(f"- Task feature: {'Yes' if p.is_task_feature else 'No'}")
        out.append("\n**Top activating prompts:**")
        for prompt, val in p.top_prompts[:5]:
            out.append(f"- [{val:.3f}] {prompt[:80]}")
    else:
        out.append("_Run 'Profile all features' first to see top prompts._")

    # Show decoder direction stats
    sae = STATE["sae"]
    if sae:
        direction = sae.get_decoder_direction(feature_idx)
        out.append(f"\n**Decoder direction norm:** {direction.norm().item():.4f}")
        out.append(f"**Top 3 decoder dims:** {direction.abs().topk(3).indices.tolist()}")

    return "\n".join(out)


def label_feature(feature_idx_str: str, label: str):
    """Save a label for a feature."""
    try:
        feature_idx = int(feature_idx_str)
    except ValueError:
        return "Invalid feature index."

    analyser = STATE["analyser"]
    labels   = STATE["labels"]

    analyser.label_feature(feature_idx, label)
    labels[feature_idx] = label

    # Auto-save
    os.makedirs("results", exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

    return f"Feature {feature_idx} labelled: '{label}' (saved)"


def profile_features():
    """Profile all features on the country-capital dataset."""
    analyser  = STATE["analyser"]
    collector = STATE["collector"]
    device    = STATE["device"]

    dataset     = json.load(open(DATASET_PATH))
    tmpl        = dataset["prompt_template"]
    task_prompts = [tmpl.format(country=p["country"]) for p in dataset["pairs"]]
    ctrl_prompts = [
        "France is a country in Western Europe",
        "Germany is known for its engineering",
        "The quick brown fox jumps over the lazy dog",
        "Photosynthesis converts sunlight into energy",
        "Water boils at one hundred degrees Celsius",
    ]

    all_prompts = task_prompts + ctrl_prompts
    profiles    = analyser.profile_all_features(all_prompts, top_n=5)
    STATE["profiles"] = profiles

    task_features = analyser.find_task_features(
        task_prompts, ctrl_prompts, top_k=10
    )
    for feat_idx, score in task_features:
        if feat_idx in profiles:
            profiles[feat_idx].is_task_feature = True

    # Build summary dataframe
    rows = []
    for p in sorted(profiles.values(),
                    key=lambda x: x.max_activation, reverse=True)[:30]:
        rows.append({
            "Feature":    p.feature_idx,
            "Label":      p.label or "(unlabelled)",
            "Max act":    round(p.max_activation, 3),
            "Freq":       f"{p.activation_freq:.1%}",
            "Task feat":  "Yes" if p.is_task_feature else "",
            "Top prompt": p.top_prompts[0][0][:60] if p.top_prompts else "",
        })

    df = pd.DataFrame(rows)
    return df, f"Profiled {len(profiles)} features. {sum(1 for p in profiles.values() if p.is_task_feature)} task-relevant features found."


# ---------------------------------------------------------------------------
# Tab 2 — Ablation lab
# ---------------------------------------------------------------------------

def run_ablation(features_str: str, method: str):
    """Run ablation experiment on specified features."""
    exp    = STATE["exp"]
    device = STATE["device"]

    # Parse feature indices
    try:
        feature_indices = [int(x.strip()) for x in features_str.split(",") if x.strip()]
    except ValueError:
        return "Invalid feature indices. Enter comma-separated integers.", None

    if not feature_indices:
        return "Enter at least one feature index.", None

    prompts, expected = build_task_pairs(DATASET_PATH)

    # Baseline
    baseline_acc, baseline_results = exp.measure_accuracy(
        prompts, expected, top_k=5, verbose=False
    )

    # Ablated
    result = exp.ablate_features(
        prompts, expected,
        feature_indices=feature_indices,
        method=method,
        verbose=False,
    )

    # Build result table
    rows = []
    for prompt, exp_cap in zip(prompts, expected):
        country = prompt.split(".")[-1].strip().rstrip(":")
        base_correct = baseline_results[prompt]
        abl_correct  = result.per_country.get(country, (False, False))[1]
        base_pred    = _get_prediction(collector=STATE["collector"],
                                       prompt=prompt, device=device)
        rows.append({
            "Country":  country,
            "Expected": exp_cap,
            "Baseline": "✓" if base_correct else "✗",
            "Ablated":  "✓" if abl_correct  else "✗",
            "Changed":  "BROKE" if (base_correct and not abl_correct) else
                        "FIXED" if (not base_correct and abl_correct) else "",
        })

    df = pd.DataFrame(rows)

    summary = (
        f"**Features ablated:** {feature_indices}  \n"
        f"**Method:** {method}  \n"
        f"**Baseline accuracy:** {result.baseline_acc:.1%}  \n"
        f"**Ablated accuracy:** {result.ablated_acc:.1%}  \n"
        f"**Accuracy drop:** {result.accuracy_drop:.1%} "
        f"({result.relative_drop:.1%} relative)  \n"
    )

    return summary, df


def run_graduated_ablation(features_str: str):
    """Run graduated ablation and return accuracy curve data."""
    exp = STATE["exp"]

    try:
        feature_indices = [int(x.strip()) for x in features_str.split(",") if x.strip()]
    except ValueError:
        return "Invalid feature indices.", None

    if not feature_indices:
        return "Enter feature indices.", None

    prompts, expected = build_task_pairs(DATASET_PATH)
    grad = exp.graduated_ablation(
        prompts, expected,
        candidate_features=feature_indices,
    )

    rows = [{"Features ablated": 0, "Accuracy": grad.baseline_acc}]
    for i, (feat, acc) in enumerate(zip(grad.feature_order, grad.accuracies[1:])):
        rows.append({
            "Features ablated": i + 1,
            "Accuracy":         acc,
            "Last feature added": feat,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Tab 3 — Country heatmap
# ---------------------------------------------------------------------------

def build_country_heatmap():
    """Build feature activation matrix across all countries."""
    analyser  = STATE["analyser"]
    collector = STATE["collector"]
    sae       = STATE["sae"]
    device    = STATE["device"]
    labels    = STATE["labels"]

    dataset   = json.load(open(DATASET_PATH))
    tmpl      = dataset["prompt_template"]
    countries = [p["country"] for p in dataset["pairs"]]

    rows = {}
    for country in countries:
        prompt = tmpl.format(country=country)
        acts   = collector.collect([prompt], seq_pos="last").to(device)
        feats  = sae.encode(acts)[0].cpu().numpy()
        rows[country] = feats

    df = pd.DataFrame(rows).T
    df.columns = [f"F{i}" for i in range(sae.cfg.dict_size)]

    # Keep only features active for at least one country
    active = df.columns[(df > 0).any()]
    df     = df[active]

    # Sort by total activation
    df = df[df.sum().sort_values(ascending=False).index[:30]]

    # Rename columns with labels
    rename = {}
    for col in df.columns:
        idx = int(col[1:])
        lbl = labels.get(idx, "")
        rename[col] = f"F{idx}" + (f" ({lbl})" if lbl else "")
    df = df.rename(columns=rename)

    return df.round(3)


# ---------------------------------------------------------------------------
# Tab 4 — Activation patching
# ---------------------------------------------------------------------------

def run_patching(src_country: str, tgt_country: str):
    """Patch source country activation into target forward pass."""
    exp      = STATE["exp"]
    collector = STATE["collector"]
    device   = STATE["device"]
    dataset  = json.load(open(DATASET_PATH))
    tmpl     = dataset["prompt_template"]

    src_prompt = tmpl.format(country=src_country)
    tgt_prompt = tmpl.format(country=tgt_country)

    # Baseline predictions
    src_top5 = collector.get_top_predictions(src_prompt, k=5)
    tgt_top5 = collector.get_top_predictions(tgt_prompt, k=5)

    # Patch
    src_tokens = collector.model.to_tokens(src_prompt, prepend_bos=True)
    tgt_tokens = collector.model.to_tokens(tgt_prompt, prepend_bos=True)

    _, src_cache = collector.model.run_with_cache(
        src_tokens,
        names_filter=exp.hook_pt,
    )
    src_act = src_cache[exp.hook_pt][0, -1, :].clone()

    def patch_hook(value, hook):
        value[0, -1, :] = src_act
        return value

    patched_logits = collector.model.run_with_hooks(
        tgt_tokens, fwd_hooks=[(exp.hook_pt, patch_hook)]
    )
    patched_pred = collector.model.to_string(
        patched_logits[0, -1, :].argmax().item()
    )
    patched_top5 = torch.softmax(patched_logits[0, -1, :], dim=-1).topk(5)
    patched_preds = [
        f"{repr(collector.model.to_string(i.item()))} ({p.item():.3f})"
        for p, i in zip(patched_top5.values, patched_top5.indices)
    ]

    changed = patched_pred.strip() != tgt_top5[0][0].strip()

    result = (
        f"## Patching: {src_country} → {tgt_country}\n\n"
        f"**Hook:** `{exp.hook_pt}`\n\n"
        f"**{src_country} baseline:** {[t for t, _ in src_top5[:3]]}\n\n"
        f"**{tgt_country} baseline:** {[t for t, _ in tgt_top5[:3]]}\n\n"
        f"**After patching {src_country} activation into {tgt_country}:**  \n"
        f"Top prediction: `{patched_pred}`  \n"
        f"Top-5: {patched_preds}\n\n"
        f"**Prediction changed:** {'Yes ← capital transferred!' if changed else 'No'}"
    )

    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_prediction(collector, prompt: str, device: str) -> str:
    tokens = collector.model.to_tokens(prompt, prepend_bos=True)
    logits = collector.model(tokens)
    return collector.model.to_string(logits[0, -1, :].argmax().item())


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app():
    """Construct and return the Gradio app."""

    with gr.Blocks(title="SAE Feature Explorer", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # SAE Feature Explorer
        **Mechanistic interpretability of GPT-2 Medium country-capital recall**
        Model: `gpt2-medium` | Hook: `hook_resid_post` layer 14 | SAE dict size: 512
        """)

        # ── Tab 1: Prompt analyser ──────────────────────────────
        with gr.Tab("Prompt analyser"):
            gr.Markdown("Enter any prompt to see which SAE features activate.")

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="France: Paris. Germany: Berlin. Japan: Tokyo. Egypt: Cairo. Italy:",
                    lines=2,
                    scale=4,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Top-k features", scale=1,
                )

            analyse_btn  = gr.Button("Analyse prompt", variant="primary")
            summary_out  = gr.Markdown()
            features_tbl = gr.Dataframe(label="Active features", interactive=False)

            gr.Markdown("### Feature detail")
            with gr.Row():
                feat_idx_input = gr.Textbox(
                    label="Feature index", placeholder="304", scale=1
                )
                feat_label_input = gr.Textbox(
                    label="Assign label", placeholder="capital query gating", scale=2
                )
                detail_btn = gr.Button("Show detail", scale=1)
                label_btn  = gr.Button("Save label", scale=1)

            feat_detail_out = gr.Markdown()
            label_status    = gr.Markdown()

            gr.Markdown("### Profile all features")
            profile_btn     = gr.Button("Run profile (takes ~30s)")
            profile_status  = gr.Markdown()
            profile_tbl     = gr.Dataframe(label="Feature profiles", interactive=False)

            # Preset prompts
            gr.Markdown("### Quick examples")
            with gr.Row():
                for country in ["France", "Japan", "China", "Russia"]:
                    gr.Button(
                        f"{country}",
                        size="sm",
                    ).click(
                        fn=lambda c=country: (
                            f"France: Paris. Germany: Berlin. Japan: Tokyo. Egypt: Cairo. {c}:",
                        ),
                        outputs=[prompt_input],
                    )

            analyse_btn.click(
                fn=analyse_prompt,
                inputs=[prompt_input, top_k_slider],
                outputs=[summary_out, features_tbl, label_status],
            )
            detail_btn.click(
                fn=get_feature_detail,
                inputs=[feat_idx_input],
                outputs=[feat_detail_out],
            )
            label_btn.click(
                fn=label_feature,
                inputs=[feat_idx_input, feat_label_input],
                outputs=[label_status],
            )
            profile_btn.click(
                fn=profile_features,
                outputs=[profile_tbl, profile_status],
            )

        # ── Tab 2: Ablation lab ─────────────────────────────────
        with gr.Tab("Ablation lab"):
            gr.Markdown("""
            Ablate specific SAE features and measure the effect on capital recall.
            **Top task features:** 304, 201, 473, 197, 294
            """)

            with gr.Row():
                ablation_feats  = gr.Textbox(
                    label="Feature indices (comma-separated)",
                    value="304",
                    placeholder="304, 201, 473",
                    scale=3,
                )
                ablation_method = gr.Radio(
                    choices=["zero", "mean"],
                    value="zero",
                    label="Method",
                    scale=1,
                )

            with gr.Row():
                ablate_btn  = gr.Button("Run ablation", variant="primary")
                grad_btn    = gr.Button("Graduated ablation")

            ablation_summary = gr.Markdown()
            ablation_tbl     = gr.Dataframe(label="Per-country results", interactive=False)
            grad_tbl         = gr.Dataframe(label="Graduated ablation curve", interactive=False)

            gr.Markdown("### Quick experiments")
            with gr.Row():
                gr.Button("Ablate F304 alone").click(
                    fn=lambda: "304", outputs=[ablation_feats]
                )
                gr.Button("Ablate top-3").click(
                    fn=lambda: "304, 201, 473", outputs=[ablation_feats]
                )
                gr.Button("Ablate top-5").click(
                    fn=lambda: "304, 201, 473, 197, 294", outputs=[ablation_feats]
                )

            ablate_btn.click(
                fn=run_ablation,
                inputs=[ablation_feats, ablation_method],
                outputs=[ablation_summary, ablation_tbl],
            )
            grad_btn.click(
                fn=run_graduated_ablation,
                inputs=[ablation_feats],
                outputs=[grad_tbl],
            )

        # ── Tab 3: Country heatmap ──────────────────────────────
        with gr.Tab("Country heatmap"):
            gr.Markdown("""
            Feature activation matrix across all countries.
            Rows = countries, columns = SAE features.
            Reveals which features are country-specific vs shared.
            """)

            heatmap_btn = gr.Button("Build heatmap", variant="primary")
            heatmap_tbl = gr.Dataframe(label="Activation matrix", interactive=False)

            heatmap_btn.click(fn=build_country_heatmap, outputs=[heatmap_tbl])

        # ── Tab 4: Activation patching ──────────────────────────
        with gr.Tab("Activation patching"):
            gr.Markdown("""
            Patch the residual stream activation from one country's prompt
            into another country's forward pass.
            If the model predicts the source country's capital → the activation
            encodes capital identity at this layer.
            """)

            countries_list = [
                "France", "Germany", "Japan", "Brazil", "Australia",
                "Egypt", "India", "Canada", "Argentina", "Nigeria",
                "Italy", "Spain", "China", "Russia", "Mexico",
            ]

            with gr.Row():
                src_dropdown = gr.Dropdown(
                    choices=countries_list, value="France",
                    label="Source country (donate activation)", scale=1,
                )
                tgt_dropdown = gr.Dropdown(
                    choices=countries_list, value="Germany",
                    label="Target country (receive activation)", scale=1,
                )

            patch_btn    = gr.Button("Run patching experiment", variant="primary")
            patch_result = gr.Markdown()

            gr.Markdown("### Quick pairs")
            pairs = [
                ("France", "Germany"),
                ("Japan", "China"),
                ("Italy", "Spain"),
                ("Egypt", "Russia"),
            ]
            with gr.Row():
                for src, tgt in pairs:
                    gr.Button(f"{src}→{tgt}", size="sm").click(
                        fn=lambda s=src, t=tgt: (s, t),
                        outputs=[src_dropdown, tgt_dropdown],
                    )

            patch_btn.click(
                fn=run_patching,
                inputs=[src_dropdown, tgt_dropdown],
                outputs=[patch_result],
            )

        # ── Tab 5: Research summary ─────────────────────────────
        with gr.Tab("Research summary"):
            gr.Markdown("""
            ## Key findings

            ### Finding 1 — Capital information lives in attention stream
            Layer-by-layer patching (France → Germany):
            - **Layers 0–13:** No effect — Germany still predicts "Berlin"
            - **Layer 14 resid_post:** First transfers capital → predicts "Paris"
            - **hook_mlp_out at any layer:** Never transfers capital identity
            - **Conclusion:** Country-capital associations are written by attention
              heads at layers 14–23, not by MLP layers

            ### Finding 2 — Single features causally gate capital recall
            | Feature | Disc. score | Accuracy drop |
            |---------|-------------|---------------|
            | F304    | 4.684       | 86.7% → 0.0% |
            | F201    | 1.886       | 86.7% → 0.0% |
            | F473    | 1.575       | 86.7% → 0.0% |
            | F197    | 1.550       | 86.7% → 0.0% |
            | F294    | 1.466       | 86.7% → 0.0% |

            Ablating **any single** top feature drops accuracy to 0%.
            Post-ablation outputs: "First", "Great", "Good", "Making" — incoherent,
            not wrong capitals. Suggests these features **gate** the recall pathway
            rather than storing individual country-capital pairs.

            ### Finding 3 — L0=149 indicates dense residual stream
            The residual stream at layer 14 activates ~149/512 features per token,
            much denser than the MLP stream (L0=3.6). The residual stream integrates
            information from all previous layers, so higher L0 is expected.

            ### Open questions
            1. Are features 304/201/473 country-specific or query-structure gates?
            2. Do these features activate on other factual recall tasks?
            3. Which attention heads write to these SAE features at layer 14?

            ---
            **Model:** GPT-2 Medium (345M) | **Layer:** 14 `hook_resid_post`
            **SAE:** dict_size=512, L0=149, zero dead features
            **Baseline accuracy:** 86.7% (13/15 countries, top-5)
            """)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models()
    app = build_app()
    app.launch(
        share=True,        # generates a public URL — works in Colab
        server_port=7860,
        show_error=True,
    )