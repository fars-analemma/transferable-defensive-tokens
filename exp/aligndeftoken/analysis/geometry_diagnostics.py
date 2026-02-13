"""Geometry diagnostics for DefensiveToken embeddings.

Analyzes norm statistics, cosine similarities, and Procrustes alignment quality
for source, transferred, and natively-optimized DefensiveTokens relative to
vocabulary embedding spaces. Produces statistics JSON and three figures:
  - PCA projection of tokens vs vocabulary
  - Norm histogram with DT vertical lines
  - Cosine similarity heatmap (transferred vs native)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(RESULTS, "figures")
OUTPUTS = os.path.join(BASE, "outputs")
DT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "DefensiveToken", "defensivetokens.json")

os.makedirs(FIGURES, exist_ok=True)

DIRECTIONS = [
    {
        "label": "Llama-3.1 -> Llama-3",
        "short": "31to3",
        "source_model": "meta-llama/Llama-3.1-8B-Instruct",
        "target_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama31_to_llama3.pt"),
        "procrustes_pt": os.path.join(OUTPUTS, "procrustes_W_llama31_to_llama3.pt"),
    },
    {
        "label": "Llama-3 -> Llama-3.1",
        "short": "3to31",
        "source_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "target_model": "meta-llama/Llama-3.1-8B-Instruct",
        "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama3_to_llama31.pt"),
        "procrustes_pt": os.path.join(OUTPUTS, "procrustes_W_llama3_to_llama31.pt"),
    },
]


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.sum(a_n * b_n, axis=-1)


def cosine_sim_matrix(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def load_vocab_embeddings(model_name):
    import glob
    from safetensors import safe_open
    print(f"  Loading vocab embeddings from {model_name} (safetensors, embedding layer only)...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    slug = model_name.replace("/", "--")
    pattern = os.path.join(cache_dir, f"models--{slug}", "snapshots", "*", "model.safetensors.index.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Cannot find safetensors index for {model_name}")
    idx_path = matches[0]
    snap_dir = os.path.dirname(idx_path)
    with open(idx_path) as f:
        idx = json.load(f)
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    shard_path = os.path.join(snap_dir, shard)
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        emb = sf.get_tensor("model.embed_tokens.weight").float().numpy().astype(np.float64)
    print(f"  Loaded: shape={emb.shape}")
    return emb


def compute_norm_stats(label, T_s, T_t, T_native, vocab_s, vocab_t):
    results = {}

    for name, arr in [("source_dt", T_s), ("transferred_dt", T_t), ("native_target_dt", T_native)]:
        l2 = np.linalg.norm(arr, axis=1)
        l1 = np.linalg.norm(arr, ord=1, axis=1)
        results[name] = {
            "l2_norms": l2.tolist(),
            "l2_mean": float(l2.mean()),
            "l2_std": float(l2.std()),
            "l1_norms": l1.tolist(),
            "l1_mean": float(l1.mean()),
            "l1_std": float(l1.std()),
        }

    for name, arr in [("source_vocab", vocab_s), ("target_vocab", vocab_t)]:
        l2 = np.linalg.norm(arr, axis=1)
        l1 = np.linalg.norm(arr, ord=1, axis=1)
        results[name] = {
            "l2_mean": float(l2.mean()),
            "l2_std": float(l2.std()),
            "l2_median": float(np.median(l2)),
            "l1_mean": float(l1.mean()),
            "l1_std": float(l1.std()),
            "l1_median": float(np.median(l1)),
        }

    tvn = results["target_vocab"]
    for dt_name in ["source_dt", "transferred_dt", "native_target_dt"]:
        dt = results[dt_name]
        dt["l2_ratio_to_vocab"] = dt["l2_mean"] / tvn["l2_mean"]
        dt["l1_ratio_to_vocab"] = dt["l1_mean"] / tvn["l1_mean"]

    return results


def compute_cosine_stats(T_s, T_t, T_native, W):
    T_s_aligned = T_s @ W

    cos_source_vs_native_raw = cosine_sim(T_s, T_native)
    cos_source_aligned_vs_native = cosine_sim(T_s_aligned, T_native)
    cos_transferred_vs_native = cosine_sim(T_t, T_native)
    cos_source_vs_transferred = cosine_sim(T_s_aligned, T_t)

    cos_matrix_transferred_vs_native = cosine_sim_matrix(T_t, T_native)

    return {
        "source_vs_native_raw": {
            "per_token": cos_source_vs_native_raw.tolist(),
            "mean": float(cos_source_vs_native_raw.mean()),
            "note": "Different embedding spaces, cosine not directly meaningful"
        },
        "source_aligned_vs_native": {
            "per_token": cos_source_aligned_vs_native.tolist(),
            "mean": float(cos_source_aligned_vs_native.mean()),
        },
        "transferred_vs_native": {
            "per_token": cos_transferred_vs_native.tolist(),
            "mean": float(cos_transferred_vs_native.mean()),
        },
        "source_aligned_vs_transferred": {
            "per_token": cos_source_vs_transferred.tolist(),
            "mean": float(cos_source_vs_transferred.mean()),
            "note": "Should be ~1.0 if T_t = T_s @ W"
        },
        "heatmap_transferred_vs_native": cos_matrix_transferred_vs_native.tolist(),
    }


def compute_procrustes_quality(vocab_s, vocab_t, W):
    Y = vocab_t.astype(np.float64)
    X = vocab_s.astype(np.float64)
    XW = X @ W
    residual = np.linalg.norm(XW - Y, "fro")
    Y_norm = np.linalg.norm(Y, "fro")
    rel_residual_pct = float(residual / Y_norm * 100)
    return {
        "residual_frobenius": float(residual),
        "Y_frobenius": float(Y_norm),
        "relative_residual_pct": rel_residual_pct,
    }


def plot_pca(all_dir_data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, ddata in enumerate(all_dir_data):
        ax = axes[idx]
        vocab_t = ddata["vocab_t"]
        T_s_aligned = ddata["T_s"] @ ddata["W"]
        T_t = ddata["T_t"]
        T_native = ddata["T_native"]

        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(vocab_t), size=5000, replace=False)
        vocab_sample = vocab_t[sample_idx]

        all_points = np.vstack([vocab_sample, T_s_aligned, T_t, T_native])
        pca = PCA(n_components=2)
        pca.fit(vocab_sample)
        proj = pca.transform(all_points)

        n_vocab = len(vocab_sample)
        n_dt = 5

        ax.scatter(proj[:n_vocab, 0], proj[:n_vocab, 1],
                   c="lightgray", s=2, alpha=0.3, label="Vocab (5k sample)", rasterized=True)
        offset = n_vocab
        ax.scatter(proj[offset:offset+n_dt, 0], proj[offset:offset+n_dt, 1],
                   c="blue", s=120, marker="^", edgecolors="black", linewidths=0.5,
                   label="Source DT (aligned)", zorder=5)
        offset += n_dt
        ax.scatter(proj[offset:offset+n_dt, 0], proj[offset:offset+n_dt, 1],
                   c="red", s=120, marker="s", edgecolors="black", linewidths=0.5,
                   label="Transferred DT", zorder=5)
        offset += n_dt
        ax.scatter(proj[offset:offset+n_dt, 0], proj[offset:offset+n_dt, 1],
                   c="green", s=120, marker="D", edgecolors="black", linewidths=0.5,
                   label="Native Target DT", zorder=5)

        ax.set_title(ddata["label"], fontsize=13)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("PCA Projection: DefensiveTokens vs Vocabulary Embeddings", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PCA plot: {save_path}")


def plot_norm_histogram(all_dir_data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, ddata in enumerate(all_dir_data):
        ax = axes[idx]
        vocab_norms = np.linalg.norm(ddata["vocab_t"], axis=1)
        source_norms = np.linalg.norm(ddata["T_s"], axis=1)
        transferred_norms = np.linalg.norm(ddata["T_t"], axis=1)
        native_norms = np.linalg.norm(ddata["T_native"], axis=1)

        ax.hist(vocab_norms, bins=100, color="lightgray", edgecolor="gray",
                alpha=0.7, label="Vocab embeddings", density=True)

        colors_src = ["#1f77b4"] * 5
        colors_tr = ["#d62728"] * 5
        colors_nat = ["#2ca02c"] * 5

        for i in range(5):
            lbl_s = "Source DT" if i == 0 else None
            lbl_t = "Transferred DT" if i == 0 else None
            lbl_n = "Native Target DT" if i == 0 else None
            ax.axvline(source_norms[i], color=colors_src[i], linestyle="--", linewidth=1.5, alpha=0.8, label=lbl_s)
            ax.axvline(transferred_norms[i], color=colors_tr[i], linestyle="-", linewidth=1.5, alpha=0.8, label=lbl_t)
            ax.axvline(native_norms[i], color=colors_nat[i], linestyle=":", linewidth=1.5, alpha=0.8, label=lbl_n)

        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Density")
        ax.set_title(ddata["label"], fontsize=13)
        ax.legend(fontsize=9)

        xmax = max(max(source_norms), max(transferred_norms), max(native_norms)) * 1.1
        ax.set_xlim(0, xmax)

    fig.suptitle("L2 Norm Distribution: Vocabulary vs DefensiveTokens", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved norm histogram: {save_path}")


def plot_cosine_heatmap(all_dir_data, all_cosine_stats, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    token_labels = [f"DT{i}" for i in range(5)]

    for idx, (ddata, cstats) in enumerate(zip(all_dir_data, all_cosine_stats)):
        ax = axes[idx]
        matrix = np.array(cstats["heatmap_transferred_vs_native"])
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=-0.2, vmax=0.5, center=0.1,
                    xticklabels=token_labels, yticklabels=token_labels)
        ax.set_xlabel("Native Target DT")
        ax.set_ylabel("Transferred DT")
        ax.set_title(f"{ddata['label']}\n(mean diag = {cstats['transferred_vs_native']['mean']:.4f})", fontsize=12)

    fig.suptitle("Cosine Similarity: Transferred vs Native DefensiveTokens", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cosine heatmap: {save_path}")


def main():
    with open(DT_PATH) as f:
        dt = json.load(f)

    model_embeds = {}
    all_dir_data = []
    all_norm_stats = []
    all_cosine_stats = []
    all_procrustes_quality = []
    full_results = {"directions": {}}

    for d in DIRECTIONS:
        print(f"\n{'='*60}")
        print(f"Direction: {d['label']}")
        print(f"{'='*60}")

        T_s = np.array(dt[d["source_model"]], dtype=np.float64)
        T_t = torch.load(d["transferred_pt"], weights_only=True).numpy().astype(np.float64)
        T_native = np.array(dt[d["target_model"]], dtype=np.float64)

        proc_data = torch.load(d["procrustes_pt"], weights_only=False)
        W = proc_data["W"].numpy().astype(np.float64)

        if d["source_model"] not in model_embeds:
            model_embeds[d["source_model"]] = load_vocab_embeddings(d["source_model"])
        if d["target_model"] not in model_embeds:
            model_embeds[d["target_model"]] = load_vocab_embeddings(d["target_model"])

        vocab_s = model_embeds[d["source_model"]]
        vocab_t = model_embeds[d["target_model"]]

        print("\n--- Norm Statistics ---")
        norms = compute_norm_stats(d["label"], T_s, T_t, T_native, vocab_s, vocab_t)
        all_norm_stats.append(norms)

        for key in ["source_dt", "transferred_dt", "native_target_dt"]:
            n = norms[key]
            print(f"  {key}: L2 mean={n['l2_mean']:.2f} +/- {n['l2_std']:.2f}, "
                  f"L1 mean={n['l1_mean']:.1f} +/- {n['l1_std']:.1f}, "
                  f"L2 ratio={n['l2_ratio_to_vocab']:.1f}x, L1 ratio={n['l1_ratio_to_vocab']:.1f}x")
        for key in ["source_vocab", "target_vocab"]:
            n = norms[key]
            print(f"  {key}: L2 mean={n['l2_mean']:.4f} +/- {n['l2_std']:.4f}, "
                  f"L1 mean={n['l1_mean']:.2f} +/- {n['l1_std']:.2f}")

        print("\n--- Cosine Similarity ---")
        cos_stats = compute_cosine_stats(T_s, T_t, T_native, W)
        all_cosine_stats.append(cos_stats)

        print(f"  Source(raw) vs Native: mean={cos_stats['source_vs_native_raw']['mean']:.4f}, "
              f"per-token={[f'{v:.4f}' for v in cos_stats['source_vs_native_raw']['per_token']]}")
        print(f"  Source(aligned) vs Native: mean={cos_stats['source_aligned_vs_native']['mean']:.4f}, "
              f"per-token={[f'{v:.4f}' for v in cos_stats['source_aligned_vs_native']['per_token']]}")
        print(f"  Transferred vs Native: mean={cos_stats['transferred_vs_native']['mean']:.4f}, "
              f"per-token={[f'{v:.4f}' for v in cos_stats['transferred_vs_native']['per_token']]}")
        print(f"  Source(aligned) vs Transferred: mean={cos_stats['source_aligned_vs_transferred']['mean']:.4f}")

        print("\n--- Procrustes Alignment Quality ---")
        pq = compute_procrustes_quality(vocab_s, vocab_t, W)
        all_procrustes_quality.append(pq)
        print(f"  ||XW* - Y||_F = {pq['residual_frobenius']:.4f}")
        print(f"  ||Y||_F = {pq['Y_frobenius']:.4f}")
        print(f"  Relative residual = {pq['relative_residual_pct']:.4f}%")

        full_results["directions"][d["label"]] = {
            "norm_stats": norms,
            "cosine_stats": cos_stats,
            "procrustes_quality": pq,
        }

        all_dir_data.append({
            "label": d["label"],
            "T_s": T_s,
            "T_t": T_t,
            "T_native": T_native,
            "W": W,
            "vocab_s": vocab_s,
            "vocab_t": vocab_t,
        })

    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    plot_pca(all_dir_data, os.path.join(FIGURES, "geometry_pca.png"))
    plot_norm_histogram(all_dir_data, os.path.join(FIGURES, "geometry_norm_histogram.png"))
    plot_cosine_heatmap(all_dir_data, all_cosine_stats, os.path.join(FIGURES, "geometry_cosine_heatmap.png"))

    json_path = os.path.join(RESULTS, "geometry_diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return full_results


if __name__ == "__main__":
    main()
