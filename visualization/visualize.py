"""
Visualization utilities for molecular generation analysis.

Provides:
  - Molecule grid images (RDKit)
  - Property distribution comparison plots
  - t-SNE / PCA latent space plots
  - Training loss curves
  - MOSES metrics bar charts
"""

from __future__ import annotations

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless safe; call plt.show() in notebooks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, QED, Descriptors, AllChem


# ---------------------------------------------------------------------------
# Molecule grid
# ---------------------------------------------------------------------------

def save_grid_image(img, path: str) -> None:
    """Save an RDKit grid image to disk, handling both return types.

    Draw.MolsToGridImage returns a PIL Image in plain Python but an
    IPython.core.display.Image (PNG bytes, no .save) inside Jupyter when
    RDKit's IPythonConsole hook is active. This helper covers both cases.
    """
    if hasattr(img, "save"):          # PIL Image
        img.save(path)
    elif hasattr(img, "data"):        # IPython Image holding raw PNG bytes
        with open(path, "wb") as f:
            f.write(img.data)
    else:
        raise TypeError(f"Cannot save image of type {type(img)}")


def draw_molecules(
    smiles_list: List[str],
    legends: List[str] | None = None,
    n_cols: int = 5,
    mol_size: tuple = (200, 200),
    save_path: str | None = None,
) -> None:
    """Draw a grid of molecules with optional per-molecule legends."""
    mols = []
    valid_legends = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            valid_legends.append(legends[i] if legends else smi[:20])

    if not mols:
        print("[viz] No valid molecules to draw.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=valid_legends,
    )
    if save_path:
        save_grid_image(img, save_path)   # robust save across environments
        print(f"[viz] Saved molecule grid → {save_path}")
    return img


# ---------------------------------------------------------------------------
# Property distribution comparison
# ---------------------------------------------------------------------------

def property_comparison(
    smiles_sets: Dict[str, List[str]],
    properties: List[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot overlapping KDE/histogram for molecular properties across sets."""
    if properties is None:
        properties = ["qed", "logp", "sa"]

    def compute_props(smiles_list: List[str]) -> pd.DataFrame:
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                q = QED.qed(mol)
                lp = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                rows.append({"qed": q, "logp": lp, "mw": mw, "sa": 0.0})
            except Exception:
                pass
        return pd.DataFrame(rows)

    n_props = len(properties)
    fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 4))
    if n_props == 1:
        axes = [axes]

    palette = sns.color_palette("husl", len(smiles_sets))
    prop_labels = {"qed": "QED", "logp": "LogP", "mw": "Mol. Weight", "sa": "SA Score"}

    for ax, prop in zip(axes, properties):
        for (name, smiles_list), color in zip(smiles_sets.items(), palette):
            df = compute_props(smiles_list[:2000])
            if prop in df.columns and len(df) > 0:
                sns.kdeplot(df[prop], ax=ax, label=name, color=color, fill=True, alpha=0.3)
        ax.set_xlabel(prop_labels.get(prop, prop))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"{prop_labels.get(prop, prop)} Distribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved property comparison → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# t-SNE / PCA of latent space
# ---------------------------------------------------------------------------

def latent_space_tsne(
    z_dict: Dict[str, np.ndarray],
    color_values: Dict[str, np.ndarray] | None = None,
    color_label: str = "QED",
    method: str = "tsne",
    save_path: str | None = None,
) -> None:
    """2-D projection of latent vectors, coloured by a property."""
    from sklearn.decomposition import PCA
    all_z = np.vstack(list(z_dict.values()))
    labels = np.concatenate(
        [np.full(len(v), k) for k, v in z_dict.items()]
    )

    # Reduce with PCA first to speed up t-SNE
    pca = PCA(n_components=min(50, all_z.shape[1]))
    z_pca = pca.fit_transform(all_z)

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
        z_2d = reducer.fit_transform(z_pca)
    else:
        z_2d = z_pca[:, :2]

    fig, axes = plt.subplots(1, 2 if color_values else 1,
                              figsize=(14 if color_values else 7, 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot 1: colour by dataset source
    palette = sns.color_palette("husl", len(z_dict))
    offset = 0
    for (name, z), color in zip(z_dict.items(), palette):
        n = len(z)
        axes[0].scatter(z_2d[offset:offset+n, 0], z_2d[offset:offset+n, 1],
                        c=[color], label=name, s=6, alpha=0.5)
        offset += n
    axes[0].legend(markerscale=2)
    axes[0].set_title(f"Latent Space ({method.upper()}) — by Source")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    # Plot 2: colour by property
    if color_values and len(axes) > 1:
        all_vals = np.concatenate(list(color_values.values()))
        sc = axes[1].scatter(z_2d[:, 0], z_2d[:, 1],
                             c=all_vals, cmap="viridis", s=6, alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label=color_label)
        axes[1].set_title(f"Latent Space — coloured by {color_label}")
        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved t-SNE plot → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories: Dict[str, List[dict]],
    metric: str = "loss",
    save_path: str | None = None,
) -> None:
    """Plot training curves for one or more models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("tab10", len(histories))
    for (name, history), color in zip(histories.items(), palette):
        epochs = [h["epoch"] for h in history]
        values = [h[metric] for h in history]
        ax.plot(epochs, values, label=name, color=color)
        if f"val_{metric}" in history[0]:
            val_values = [h[f"val_{metric}"] for h in history]
            ax.plot(epochs, val_values, "--", color=color, label=f"{name} (val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Training {metric.replace('_', ' ').title()}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved training curve → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# MOSES metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    metrics_dict: Dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Bar chart comparing MOSES metrics across models."""
    keys_to_plot = ["validity", "uniqueness", "novelty",
                    "internal_diversity", "qed_mean"]
    models = list(metrics_dict.keys())
    n_metrics = len(keys_to_plot)

    x = np.arange(n_metrics)
    width = 0.8 / len(models)
    palette = sns.color_palette("husl", len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(models, palette)):
        vals = [metrics_dict[model_name].get(k, 0.0) for k in keys_to_plot]
        ax.bar(x + i * width, vals, width, label=model_name, color=color)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys_to_plot])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (MOSES Metrics)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved metrics comparison → {save_path}")
    plt.show()
