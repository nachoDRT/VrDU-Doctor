import itertools
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_data(path: str | None = None):
    """Return L2‑normalised embeddings and string labels.

    CSV layout:
        • Columns 0‑1023 → embedding (float)
        • Column   1024  → label      (str)
    """
    if path is None:
        path = join(dirname(dirname(__file__)), "data", "de_Rodrigo_merit_secret_all_embeddings.csv")

    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1024), dtype=float)
    y = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1024, dtype=str)

    if x.shape[0] != y.shape[0]:
        raise ValueError("Embeddings and labels have different lengths")

    x /= np.linalg.norm(x, axis=1, keepdims=True)  # L2 normalise rows
    return x, y


# ---------------------------------------------------------------------------
# 2. Violin plot per dimension
# ---------------------------------------------------------------------------


def violin_per_dimension(x: np.ndarray, title: str = "Per‑dimension distribution (violin)") -> None:
    """Draw one violin plot per embedding dimension."""
    dims = x.shape[1]
    fig, ax = plt.subplots(figsize=(max(10, dims / 50), 4))

    ax.violinplot([x[:, i] for i in range(dims)], positions=np.arange(dims), showmedians=True, widths=0.8)
    ax.set(title=title, xlabel="Embedding dimension", ylabel="Value")

    step = max(1, dims // 16)
    ax.set_xticks(np.arange(0, dims, step))
    ax.set_xticklabels(np.arange(0, dims, step), rotation=90, fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Generic helpers: stats + histogram
# ---------------------------------------------------------------------------


def _print_stats(arr: np.ndarray):
    print("Mean   :", arr.mean())
    print("Std    :", arr.std())
    for p in (5, 50, 95, 99):
        print(f"Percentile {p:>2}:", np.percentile(arr, p))


def _plot_hist(vals: np.ndarray, title: str, xlabel: str):
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Cosine / angular metrics + inter‑group stats
# ---------------------------------------------------------------------------


def cosine_stats(x: np.ndarray, tag: str = "") -> np.ndarray:
    sim = cosine_similarity(x)
    vals = sim[np.triu_indices_from(sim, k=1)]
    print(f"\n=== Cosine similarity {tag} ===")
    _print_stats(vals)
    return vals


def angle_stats_from_cos(cos_vals: np.ndarray, tag: str = "") -> np.ndarray:
    ang = np.degrees(np.arccos(np.clip(cos_vals, -1.0, 1.0)))
    print(f"\n=== Angular distance (°) {tag} ===")
    _print_stats(ang)
    return ang


def pairwise_group_stats(x: np.ndarray, labels: np.ndarray[str], prefix: str = "", plot: bool = True):
    """Print cosine stats (and histograms) for every pair of label groups."""
    for a, b in itertools.combinations(np.unique(labels), 2):
        sims = cosine_similarity(x[labels == a], x[labels == b]).ravel()
        tag = f"{prefix}[{a} vs {b}]"
        print(f"\n--- Cosine similarity {tag} ---")
        _print_stats(sims)
        if plot:
            _plot_hist(sims, f"Cosine {tag}", "Cosine sim")
            ang = np.degrees(np.arccos(np.clip(sims, -1.0, 1.0)))
            _plot_hist(ang, f"Angle (°) {tag}", "Angle (deg)")


# ---------------------------------------------------------------------------
# 5. PCA utilities, scatter, tangential projection
# ---------------------------------------------------------------------------


def pca_reduce(x: np.ndarray, n_comp: int = 2, l2: bool = True):
    """Reduce *x* to *n_comp* principal components.

    Returns
    -------
    pcs : np.ndarray
        Reduced embeddings (optionally L2‑normalised).
    explained : float
        Cumulative explained variance ratio.
    """
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(x)
    if l2:
        pcs /= np.linalg.norm(pcs, axis=1, keepdims=True)
    explained = pca.explained_variance_ratio_.sum()
    return pcs, explained


def scatter_2d(points: np.ndarray, labels: np.ndarray[str], title: str, xlab: str = "PC1", ylab: str = "PC2"):
    uniq, inv = np.unique(labels, return_inverse=True)
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(points[:, 0], points[:, 1], c=inv, cmap="tab10", s=40, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    ax = plt.gca()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2
    ax.set_xlim(midx - half, midx + half)
    ax.set_ylim(midy - half, midy + half)

    handles, _ = sc.legend_elements()
    ax.legend(handles, uniq, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def tangential_embeddings(x: np.ndarray) -> np.ndarray:
    """Remove PC1 and renormalise to unit length."""
    u = PCA().fit(x).components_[0]
    t = x - (x @ u[:, None]) * u
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------


def main():
    """Run full analysis pipeline."""
    x, y = load_data()

    # 1. Violin plot
    violin_per_dimension(x)

    # 2. Global metrics
    cos_global = cosine_stats(x, "(global)")
    angle_stats_from_cos(cos_global, "(global)")

    # 3. PCA scatter (2‑D) with explained variance
    pcs2, exp2 = pca_reduce(x, 2)
    print(f"Explained variance (2 PC): {exp2:.2%}")
    scatter_2d(pcs2, y, "PCA 2‑D · global")

    # 4. Tangential space analysis
    t = tangential_embeddings(x)
    cos_t = cosine_stats(t, "(tangential)")
    angle_stats_from_cos(cos_t, "(tangential)")

    pcs2_t, exp2_t = pca_reduce(t, 2)
    print(f"Explained variance tangential (2 PC): {exp2_t:.2%}")
    scatter_2d(pcs2_t, y, "PCA 2‑D · tangential")

    # 5. Progressive PCA reduction loop
    print("\n=== Progressive PCA reduction ===")
    for ratio in np.arange(1.0, 0.0, -0.1):
        k = max(2, min(int(x.shape[1] * ratio), int(x.shape[0] * ratio)))
        pcs_k, explained_k = pca_reduce(x, n_comp=k)
        print(f"\n>>> {k} components — explained variance: {explained_k:.2%}")
        cosine_stats(pcs_k, f"({k} comp)")


if __name__ == "__main__":
    main()
