"""
E-Commerce Recommendation System — Ray Data Preprocessing Pipeline
===================================================================

Stage 1 of 3:  Product Data Preprocessing

What this does
--------------
1. Generates a synthetic product catalog (name, category, description, image).
2. Loads it into Ray Data.
3. Runs two parallel preprocessing branches:
   - Images  : resize → normalise → store as float32 bytes
   - Text    : clean → prepare training text
4. Writes a single preprocessed Parquet file to data/preprocessed/ ready for
   Stage 2 (Ray Train fine-tuning).

Run locally
-----------
    python preprocess_pipeline.py

Run as an Anyscale Job
----------------------
    anyscale job submit -f job_preprocess.yaml

References
----------
- Ray Data Transforming Data: https://docs.ray.io/en/latest/data/transforming-data.html
- Ray Data map_batches:       https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html
"""

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import ray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve to absolute path so Ray worker tasks write to the workspace dir,
# not to their own temp working_dir in /tmp/ray/...
OUTPUT_DIR = os.path.abspath("data/preprocessed")
BATCH_SIZE = 8          # Small – we only have ~34 products
NUM_CPUS_PER_TASK = 1
SAMPLE_LIMIT = None     # Set to an int to cap rows during development


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 0. Init Ray
    # ------------------------------------------------------------------
    ray.init(ignore_reinit_error=True)

    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = True

    print("=" * 60)
    print("E-COMMERCE PREPROCESSING PIPELINE")
    print("=" * 60)

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Generate synthetic catalog and load into Ray Data
    # ------------------------------------------------------------------
    print("\n[1/4] Generating synthetic product catalog …")

    # Import here so Ray workers can also import utils from the working dir
    from utils import generate_catalog, PRODUCTS

    records = generate_catalog(products=PRODUCTS, output_dir=os.path.abspath("data/raw"))

    ds = ray.data.from_items(records)

    if SAMPLE_LIMIT:
        ds = ds.limit(SAMPLE_LIMIT)

    print(f"  Rows : {ds.count()}")
    print(f"  Schema: {ds.schema()}")

    # ------------------------------------------------------------------
    # 2. Preprocess images
    # ------------------------------------------------------------------
    print("\n[2/4] Preprocessing images (resize + normalise) …")

    from utils import preprocess_image_batch

    ds = ds.map_batches(
        preprocess_image_batch,
        batch_size=BATCH_SIZE,
        num_cpus=NUM_CPUS_PER_TASK,
        batch_format="numpy",
    )

    # ------------------------------------------------------------------
    # 3. Preprocess text
    # ------------------------------------------------------------------
    print("\n[3/4] Preprocessing text (clean) …")

    from utils import preprocess_text_batch

    ds = ds.map_batches(
        preprocess_text_batch,
        batch_size=BATCH_SIZE,
        num_cpus=NUM_CPUS_PER_TASK,
        batch_format="numpy",
    )

    # ------------------------------------------------------------------
    # 4. Write to Parquet
    # ------------------------------------------------------------------
    print(f"\n[4/4] Writing preprocessed data to '{OUTPUT_DIR}' …")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ds.write_parquet(OUTPUT_DIR)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    result_ds = ray.data.read_parquet(OUTPUT_DIR)
    print(f"  Rows written : {result_ds.count()}")
    print(f"  Schema       : {result_ds.schema()}")
    print(f"  Wall time    : {elapsed:.1f}s")
    print(f"  Output dir   : {OUTPUT_DIR}")

    sample = result_ds.take(2)
    print("\nSample records:")
    for row in sample:
        print(f"  [{row['product_id']}] {row['name']!r:30s}  "
              f"category={row['category']!r:18s}  "
              f"img_tensor_bytes={len(row['image_tensor_bytes'])} bytes  "
              f"text_clean length={len(row['text_clean'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
