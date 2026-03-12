"""Visualization utilities for embedding analysis."""

from collections import Counter
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot_category_samples(products: List[Dict], categories: List[str], get_image_fn: Callable) -> None:
    """Show one sample product image per category."""
    fig, axes = plt.subplots(1, len(categories), figsize=(15, 3))
    for ax, cat in zip(axes, categories):
        product = next(p for p in products if p["category"] == cat)
        img = get_image_fn(product)
        ax.imshow(img)
        ax.set_title(cat, fontsize=9)
        ax.axis("off")
    plt.suptitle("One sample image per category", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_products_per_category(products: List[Dict]) -> None:
    """Horizontal bar chart of product counts per category."""
    cats = [p["category"] for p in products]
    counts = sorted(Counter(cats).items())
    plt.barh(*zip(*counts))
    plt.xlabel("Products")
    plt.title("Products per category")
    plt.tight_layout()
    plt.show()


def plot_training_loss(metrics_dataframe) -> None:
    """Line plot of training loss across epochs."""
    m = metrics_dataframe
    plt.plot(m["epoch"] + 1, m["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


def plot_recommendations(
    query_img: np.ndarray,
    recommendations: List[Dict],
    products: List[Dict],
    caption: str,
    get_image_fn: Callable,
) -> None:
    """Show query image alongside recommended products."""
    fig, axes = plt.subplots(1, len(recommendations) + 1, figsize=(18, 3))

    axes[0].imshow(query_img)
    axes[0].set_title("Query image", fontsize=9, color="navy", fontweight="bold")
    axes[0].axis("off")

    all_prods = {p["name"]: p for p in products}
    for ax, rec in zip(axes[1:], recommendations):
        prod_name = rec["name"]
        prod = all_prods.get(prod_name, {"name": prod_name, "category": rec["category"]})
        rec_img = get_image_fn({**prod, "category": rec["category"]})
        ax.imshow(rec_img)
        ax.set_title(
            f"{rec['rank']}. {rec['name'][:18]}\n{rec['similarity']:.2f}", fontsize=8
        )
        ax.axis("off")

    plt.suptitle(f'Caption: "{caption}"', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(
    embeddings, labels, n=10, title="Product embedding similarity (top 10)"
):
    """Plot cosine similarity heatmap for the first n embeddings."""
    n = min(n, len(embeddings))
    sub = embeddings[:n]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sim = (sub / norms) @ (sub / norms).T

    short_labels = [lbl[:20] for lbl in labels[:n]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_tsne_comparison(base_embs, tuned_embs, labels):
    """Side-by-side t-SNE of base vs fine-tuned embeddings, coloured by category."""
    from sklearn.manifold import TSNE

    cat_list = sorted(set(labels))
    colors = [cat_list.index(c) for c in labels]
    cmap = plt.cm.get_cmap("tab10", len(cat_list))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, embs, title in [
        (axes[0], base_embs, "Base model (all-MiniLM-L6-v2)"),
        (axes[1], tuned_embs, "Fine-tuned (contrastive loss)"),
    ]:
        xy = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(embs)
        ax.scatter(
            xy[:, 0], xy[:, 1], c=colors, cmap=cmap, s=80, vmin=0, vmax=len(cat_list)
        )
        ax.set_title(title)
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=9,
            label=c,
        )
        for i, c in enumerate(cat_list)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(cat_list), fontsize=9)
    plt.suptitle("t-SNE: product embeddings coloured by category", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def compute_category_precision_at_5(embeddings, metadata):
    """Return fraction of top-5 neighbors sharing the same category (macro-averaged)."""
    n = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    cats = [m["category"] for m in metadata]
    return np.mean(
        [
            sum(cats[j] == cats[i] for j in np.argsort(sim[i])[-5:]) / 5
            for i in range(len(embeddings))
        ]
    )
