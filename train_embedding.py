"""
E-Commerce Recommendation System — Ray Train Embedding Fine-Tuning
==================================================================

Stage 2 of 3:  Fine-tune a text embedding model

What this does
--------------
1. Reads preprocessed product data from data/preprocessed/ (Parquet).
2. Builds contrastive pairs: texts from the same category are "similar".
3. Fine-tunes sentence-transformers/all-MiniLM-L6-v2 using cosine-similarity
   loss (CosineSimilarityLoss from sentence-transformers).
4. Saves the fine-tuned model to models/embedding_model/.
5. Encodes all products and saves:
   - models/product_embeddings.npy  (float32 array, shape [N, 384])
   - models/product_metadata.json   (list of {product_id, name, category})

Run locally
-----------
    python train_embedding.py

Run as an Anyscale Job
----------------------
    anyscale job submit -f job_train.yaml

Ray version: 2.x  (tested with Ray ≥ 2.20)
Base image:  anyscale/ray:2.47.1-slim-py312   (CPU-only, Python 3.12)
See https://docs.anyscale.com/reference/base-images for latest images.

References
----------
- Ray Train TorchTrainer:  https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
- Ray Train user guides:   https://docs.ray.io/en/latest/train/user-guides.html
- Sentence-transformers:   https://www.sbert.net/
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import ray
import ray.data
import torch
import torch.nn.functional as F
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
    get_checkpoint,
    get_context,
    report,
)
from ray.train.torch import TorchTrainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use absolute paths so Ray Train workers (which have different working dirs)
# read/write from the workspace directory, not their temp working_dir in /tmp/ray/...
_HERE = os.path.abspath(os.path.dirname(__file__)) if "__file__" in dir() else os.getcwd()
PREPROCESSED_DIR = os.path.join(_HERE, "data", "preprocessed")
MODEL_OUTPUT_DIR = os.path.join(_HERE, "models", "embedding_model")
EMBEDDINGS_PATH  = os.path.join(_HERE, "models", "product_embeddings.npy")
METADATA_PATH    = os.path.join(_HERE, "models", "product_metadata.json")

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, CPU-fast
EMBEDDING_DIM = 384    # output size of all-MiniLM-L6-v2
MAX_SEQ_LEN = 128

NUM_EPOCHS = 5
BATCH_SIZE = 8          # small dataset: 34 products ≈ ~50 pairs
LEARNING_RATE = 2e-5
SEED = 42

TRAIN_RESULT_DIR = os.path.join(_HERE, "models", "ray_train_results")


# ---------------------------------------------------------------------------
# Contrastive dataset
# ---------------------------------------------------------------------------

class ContrastivePairDataset(Dataset):
    """
    Generates (anchor, positive, label=1.0) pairs from products in the same
    category, and (anchor, negative, label=-1.0) pairs from different
    categories.  Labels match CosineSimilarityLoss expectations.
    """

    def __init__(self, records: List[dict], neg_ratio: float = 0.5, seed: int = SEED):
        self.records = records
        rng = np.random.default_rng(seed)

        # Group indices by category
        cat_to_idx: dict[str, list] = {}
        for i, r in enumerate(records):
            cat_to_idx.setdefault(r["category"], []).append(i)

        pairs: List[Tuple[str, str, float]] = []

        # Positives: all within-category pairs
        for indices in cat_to_idx.values():
            for i, a in enumerate(indices):
                for b in indices[i + 1 :]:
                    pairs.append((records[a]["text_clean"], records[b]["text_clean"], 1.0))

        # Negatives: random cross-category pairs (at most neg_ratio × positives)
        cats = list(cat_to_idx.keys())
        n_neg = max(1, int(len(pairs) * neg_ratio))
        for _ in range(n_neg):
            cat_a, cat_b = rng.choice(cats, size=2, replace=False)
            ia = rng.choice(cat_to_idx[cat_a])
            ib = rng.choice(cat_to_idx[cat_b])
            pairs.append((records[ia]["text_clean"], records[ib]["text_clean"], -1.0))

        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return a, b, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop (runs inside each Ray Train worker)
# ---------------------------------------------------------------------------

def _forward_embeddings(model, texts: list, device: str) -> torch.Tensor:
    """Run a SentenceTransformer forward pass with gradient tracking."""
    features = model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items()}
    out = model(features)
    return out["sentence_embedding"]


def train_loop_per_worker(config: dict) -> None:
    """Fine-tune the embedding model on each distributed worker."""
    from sentence_transformers import SentenceTransformer

    rank = get_context().get_world_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Records are passed via config (dataset is small enough)
    records = config["records"]

    if rank == 0:
        print(f"Worker {rank}: {len(records)} product records")

    # ------------------------------------------------------------------
    # Build contrastive pairs
    # ------------------------------------------------------------------
    pair_dataset = ContrastivePairDataset(records, seed=config["seed"])

    if rank == 0:
        print(f"  Contrastive pairs: {len(pair_dataset)} "
              f"(pos+neg within/across categories)")

    def collate(batch):
        texts_a, texts_b, labels = zip(*batch)
        return list(texts_a), list(texts_b), torch.stack(labels)

    loader = DataLoader(
        pair_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    # ------------------------------------------------------------------
    # Load sentence-transformer model
    # ------------------------------------------------------------------
    model = SentenceTransformer(config["base_model"], device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )

    # Resume from checkpoint if available
    start_epoch = 0
    ckpt = get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            meta = torch.load(os.path.join(ckpt_dir, "meta.pt"), map_location="cpu")
            start_epoch = meta.get("epoch", 0) + 1
            model = SentenceTransformer(ckpt_dir, device=device)
        if rank == 0:
            print(f"Resumed from checkpoint at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\nFine-tuning {config['base_model']} on {device}")
        print(f"Epochs: {config['epochs']}  |  Batch size: {config['batch_size']}")
        print(f"Pairs per epoch: {len(pair_dataset)}")
        print("-" * 50)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for texts_a, texts_b, labels in loader:
            labels = labels.to(device)

            # Use forward() directly so gradients flow (encode() uses no_grad)
            emb_a = _forward_embeddings(model, texts_a, device)
            emb_b = _forward_embeddings(model, texts_b, device)

            cos_sim = F.cosine_similarity(emb_a, emb_b)
            loss = F.mse_loss(cos_sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        if rank == 0:
            print(f"Epoch {epoch + 1:2d}/{config['epochs']}  loss={avg_loss:.4f}")

        # Checkpoint on rank-0 only
        with tempfile.TemporaryDirectory() as tmpdir:
            if rank == 0:
                model.save(tmpdir)
                torch.save({"epoch": epoch}, os.path.join(tmpdir, "meta.pt"))
                ckpt_out = Checkpoint.from_directory(tmpdir)
            else:
                ckpt_out = None

            report({"epoch": epoch, "train_loss": avg_loss}, checkpoint=ckpt_out)


# ---------------------------------------------------------------------------
# Driver: launch Ray Train
# ---------------------------------------------------------------------------

def run_training() -> ray.train.Result:
    """Configure and launch TorchTrainer."""
    print("\n" + "=" * 60)
    print("STAGE 2 — RAY TRAIN: EMBEDDING FINE-TUNING")
    print("=" * 60)

    # Load preprocessed data with pandas (34 rows — small enough to pass via config)
    print(f"\n[1/3] Loading preprocessed data from '{PREPROCESSED_DIR}' …")
    import glob as _glob
    import pandas as _pd
    parquet_files = sorted(_glob.glob(f"{PREPROCESSED_DIR}/*.parquet"))
    df = _pd.concat([_pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    records = df[["product_id", "name", "category", "text_clean"]].to_dict(orient="records")
    print(f"  Rows loaded: {len(records)}")

    train_config = {
        "base_model": BASE_MODEL,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "seed": SEED,
        "records": records,   # pass directly — dataset is tiny
    }

    print("\nTraining config:")
    for k, v in train_config.items():
        print(f"  {k}: {v}")

    print("\n[2/3] Launching TorchTrainer …")

    Path(TRAIN_RESULT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=1,     # single worker: dataset is tiny
            use_gpu=torch.cuda.is_available(),
        ),
        run_config=RunConfig(
            name="ecomm_embedding_finetune",
            storage_path=os.path.abspath(TRAIN_RESULT_DIR),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="train_loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(max_failures=1),
        ),
    )

    result = trainer.fit()

    print("\nTraining complete!")
    best_ckpt = result.best_checkpoints[0][0] if result.best_checkpoints else None
    print(f"  Best checkpoint: {best_ckpt}")
    return result


# ---------------------------------------------------------------------------
# Post-training: export model + compute embeddings
# ---------------------------------------------------------------------------

def export_model_and_embeddings(result: ray.train.Result) -> None:
    """Save the fine-tuned model and pre-compute all product embeddings."""
    from sentence_transformers import SentenceTransformer

    print("\n[3/3] Exporting model and computing product embeddings …")

    Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # Restore best model from checkpoint
    checkpoint = result.best_checkpoints[0][0]
    with checkpoint.as_directory() as ckpt_dir:
        model = SentenceTransformer(ckpt_dir)
        model.save(MODEL_OUTPUT_DIR)
    print(f"  Model saved to: {MODEL_OUTPUT_DIR}")

    # Load all products for inference
    ds = ray.data.read_parquet(PREPROCESSED_DIR)
    records = ds.select_columns(
        ["product_id", "name", "category", "text_clean"]
    ).take_all()

    texts = [r["text_clean"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"  Embeddings saved to: {EMBEDDINGS_PATH}  shape={embeddings.shape}")

    metadata = [
        {"product_id": r["product_id"], "name": r["name"], "category": r["category"]}
        for r in records
    ]
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {METADATA_PATH}  ({len(metadata)} products)")

    # Quick sanity-check: nearest neighbour for first product
    print("\nSanity check – nearest neighbours for first product:")
    q_emb = embeddings[0]
    sims = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
    )
    top5 = np.argsort(sims)[::-1][:5]
    for rank_i, idx in enumerate(top5):
        m = metadata[idx]
        print(f"  {rank_i+1}. [{m['category']:18s}] {m['name']}  (sim={sims[idx]:.3f})")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ray.init(ignore_reinit_error=True)

    t0 = time.time()
    result = run_training()
    export_model_and_embeddings(result)

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"  Wall time      : {time.time() - t0:.1f}s")
    print(f"  Model          : {MODEL_OUTPUT_DIR}")
    print(f"  Embeddings     : {EMBEDDINGS_PATH}")
    print(f"  Metadata       : {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
