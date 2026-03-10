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
import time
from pathlib import Path

import numpy as np
import ray
import ray.data
import torch
from ray.train import (
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from utils.training import ContrastivePairDataset, train_loop_per_worker  # noqa: F401

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

NUM_EPOCHS = 2
BATCH_SIZE = 8          # small dataset: 34 products ≈ ~50 pairs
LEARNING_RATE = 2e-5
SEED = 42

# Use shared cluster storage if available (required for multi-node Ray Train jobs),
# otherwise fall back to a local path for single-node development.
if os.path.isdir("/mnt/cluster_storage"):
    TRAIN_RESULT_DIR = "/mnt/cluster_storage/ecomm_ray_train_results"
else:
    TRAIN_RESULT_DIR = os.path.join(_HERE, "models", "ray_train_results")


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
