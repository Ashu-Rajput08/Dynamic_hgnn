"""Plotting helpers shared by the notebooks.

Every function takes plain pandas/numpy/torch objects and returns a
matplotlib ``Figure`` (notebooks call ``plt.show()`` or just let the figure
display). Kept dependency-light: matplotlib + seaborn + networkx + sklearn
(already required elsewhere in the project).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .incidence import EDGE_TYPE_NAMES, HyperedgeBatch


# ---------------------------------------------------------------------------
# Raw data / spatial structure
# ---------------------------------------------------------------------------

def plot_cell_positions_3d(
    positions: pd.DataFrame,
    color: Optional[Sequence] = None,
    title: str = "Cell positions",
    cmap: str = "tab20",
) -> plt.Figure:
    """3D scatter of cell positions. ``positions`` needs columns x, y, z."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(positions["x"], positions["y"], positions["z"], c=color, cmap=cmap, s=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    if color is not None and not isinstance(color, str):
        fig.colorbar(sc, ax=ax, shrink=0.6, label="cluster")
    fig.tight_layout()
    return fig


def plot_dbscan_clusters(positions: pd.DataFrame, labels: np.ndarray, title: str = "DBSCAN clusters") -> plt.Figure:
    """Scatter of (x, y) coloured by DBSCAN cluster id; noise (-1) shown in grey."""
    fig, ax = plt.subplots(figsize=(6, 6))
    is_noise = labels == -1
    ax.scatter(positions.loc[is_noise, "x"], positions.loc[is_noise, "y"], c="lightgrey", s=12, label="noise")
    ax.scatter(
        positions.loc[~is_noise, "x"],
        positions.loc[~is_noise, "y"],
        c=labels[~is_noise],
        cmap="tab20",
        s=18,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hyperedge / incidence structure
# ---------------------------------------------------------------------------

def plot_edge_type_counts(batch: HyperedgeBatch, title: str = "Hyperedges by type") -> plt.Figure:
    """Bar chart of how many hyperedges of each type are in H_aug(t)."""
    counts = [int((batch.edge_type == i).sum().item()) for i in range(len(EDGE_TYPE_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(EDGE_TYPE_NAMES, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("# hyperedges")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_incidence_sparsity(batch: HyperedgeBatch, title: str = "H_aug(t) sparsity") -> plt.Figure:
    """Scatter plot of nonzero (cell, hyperedge) incidences, coloured by edge type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cell_idx = batch.cell_index.numpy()
    edge_idx = batch.edge_index.numpy()
    etype = batch.edge_type[batch.edge_index].numpy()

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, name in enumerate(EDGE_TYPE_NAMES):
        m = etype == i
        ax.scatter(edge_idx[m], cell_idx[m], s=1, color=colors[i], label=name)

    ax.set_xlabel("hyperedge index")
    ax.set_ylabel("cell index")
    ax.set_title(title)
    ax.legend(markerscale=5)
    fig.tight_layout()
    return fig


def plot_hyperedge_size_distribution(batch: HyperedgeBatch, title: str = "Hyperedge size distribution") -> plt.Figure:
    sizes = batch.edge_sizes().numpy()
    etype = batch.edge_type.numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, name in enumerate(EDGE_TYPE_NAMES):
        s = sizes[etype == i]
        if len(s) > 0:
            ax.hist(s, bins=range(2, int(s.max()) + 2), alpha=0.6, label=name, color=colors[i])
    ax.set_xlabel("hyperedge size (# member cells)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_attention_for_cell(
    batch: HyperedgeBatch,
    e2v_attn: torch.Tensor,
    cell_idx: int,
    cell_name: str,
    top_k: int = 15,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of a Hyper-SAGNN layer's E2V attention weights for one cell.

    ``e2v_attn`` is the per-incidence attention tensor returned by
    ``HyperSAGNNLayer(..., return_attention=True)``, aligned with
    ``batch.cell_index`` / ``batch.edge_index``.
    """
    mask = batch.cell_index.numpy() == cell_idx
    edges = batch.edge_index.numpy()[mask]
    weights = e2v_attn.numpy()[mask]
    etypes = batch.edge_type[batch.edge_index].numpy()[mask]

    order = np.argsort(-weights)[:top_k]
    edges, weights, etypes = edges[order], weights[order], etypes[order]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bar_colors = [colors[e] for e in etypes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(weights)), weights, color=bar_colors)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"e{e}\n({EDGE_TYPE_NAMES[t]})" for e, t in zip(edges, etypes)], fontsize=7, rotation=45)
    ax.set_ylabel("E2V attention weight")
    ax.set_title(title or f"Top-{top_k} hyperedge attention weights for cell '{cell_name}'")
    fig.tight_layout()
    return fig


def plot_hypergraph_subgraph(
    batch: HyperedgeBatch,
    cell_names: Sequence[str],
    cell_indices: Sequence[int],
    max_edges: int = 60,
    title: str = "Hypergraph (bipartite view)",
) -> plt.Figure:
    """Bipartite cell<->hyperedge graph for a small subset of cells."""
    keep_cells = set(cell_indices)
    mask = np.isin(batch.cell_index.numpy(), list(keep_cells))
    edge_ids = np.unique(batch.edge_index.numpy()[mask])[:max_edges]

    G = nx.Graph()
    for c in keep_cells:
        G.add_node(("cell", c), bipartite=0, label=cell_names[c])
    for e in edge_ids:
        G.add_node(("edge", int(e)), bipartite=1)

    cell_idx = batch.cell_index.numpy()
    edge_idx = batch.edge_index.numpy()
    for c, e in zip(cell_idx, edge_idx):
        if c in keep_cells and e in edge_ids:
            G.add_edge(("cell", int(c)), ("edge", int(e)))

    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    cell_nodes = [n for n in G.nodes if n[0] == "cell"]
    edge_nodes = [n for n in G.nodes if n[0] == "edge"]
    nx.draw_networkx_nodes(G, pos, nodelist=cell_nodes, node_color="#4C72B0", node_size=300, ax=ax, label="cells")
    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, node_color="#C44E52", node_size=80, ax=ax, label="hyperedges")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_labels(G, pos, labels={n: cell_names[n[1]] for n in cell_nodes}, font_size=7, ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Embeddings / training curves
# ---------------------------------------------------------------------------

def plot_embeddings_pca(
    embeddings: torch.Tensor,
    labels: Sequence,
    label_names: Optional[Sequence[str]] = None,
    title: str = "Node embeddings (PCA)",
) -> plt.Figure:
    emb = embeddings.detach().cpu().numpy()
    if emb.shape[1] > 2:
        emb = PCA(n_components=2).fit_transform(emb)

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if label_names is not None:
        handles, _ = sc.legend_elements()
        ax.legend(handles, label_names, loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_training_curves(history: pd.DataFrame, title: str = "Training curves") -> plt.Figure:
    """``history`` has one row per (epoch, t) step with loss/metric columns."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for col in ["total", "fate", "alignment", "anastomosis"]:
        if col in history.columns:
            axes[0].plot(history[col].to_numpy(), label=col, alpha=0.8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Losses")
    axes[0].legend()

    for col in ["fate_acc", "alignment_f1", "anastomosis_f1"]:
        if col in history.columns:
            axes[1].plot(history[col].to_numpy(), label=col, alpha=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Metrics")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spatial_evolution(
    counts_by_t: pd.DataFrame,
    title: str = "Hypergraph evolution over time",
) -> plt.Figure:
    """``counts_by_t`` indexed by t with columns = edge type names -> counts."""
    fig, ax = plt.subplots(figsize=(9, 4))
    counts_by_t.plot(ax=ax)
    ax.set_xlabel("timepoint t")
    ax.set_ylabel("# hyperedges")
    ax.set_title(title)
    fig.tight_layout()
    return fig
