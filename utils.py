"""
Shared utilities for the e-commerce recommendation system demo.
Kept in a single .py file so both notebooks and scripts can import it.
"""

import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]

CATEGORY_COLORS = {
    "Electronics":   (70,  130, 180),   # steel blue
    "Clothing":      (220,  90, 120),   # rose
    "Books":         (120, 170,  60),   # olive green
    "Home & Garden": (200, 150,  50),   # golden
    "Sports":        (80,  180, 140),   # teal
}

PRODUCTS = [
    # Electronics
    {"name": "Wireless Headphones",   "category": "Electronics",   "price": 79.99,  "description": "Over-ear noise-cancelling Bluetooth headphones with 30-hour battery life."},
    {"name": "Mechanical Keyboard",   "category": "Electronics",   "price": 99.99,  "description": "Compact TKL mechanical keyboard with tactile switches and RGB backlight."},
    {"name": "USB-C Hub",             "category": "Electronics",   "price": 39.99,  "description": "7-in-1 USB-C hub with HDMI, SD card, and 100W power delivery."},
    {"name": "Webcam 1080p",          "category": "Electronics",   "price": 59.99,  "description": "Full HD webcam with built-in microphone for video conferencing."},
    {"name": "Portable SSD 1TB",      "category": "Electronics",   "price": 109.99, "description": "Ultra-fast portable SSD with USB 3.2 Gen 2 and rugged aluminum case."},
    {"name": "Smart Watch",           "category": "Electronics",   "price": 199.99, "description": "Fitness smart watch with heart-rate monitor and 7-day battery."},
    {"name": "Bluetooth Speaker",     "category": "Electronics",   "price": 49.99,  "description": "Waterproof portable speaker with 360° sound and 12-hour battery."},
    {"name": "Laptop Stand",          "category": "Electronics",   "price": 29.99,  "description": "Adjustable aluminum laptop stand compatible with 10-16 inch laptops."},
    # Clothing
    {"name": "Running Shoes",         "category": "Clothing",      "price": 89.99,  "description": "Lightweight breathable running shoes with cushioned sole and wide toe box."},
    {"name": "Merino Wool Sweater",   "category": "Clothing",      "price": 69.99,  "description": "Soft and warm 100% merino wool crewneck sweater, machine washable."},
    {"name": "Waterproof Jacket",     "category": "Clothing",      "price": 129.99, "description": "Lightweight packable rain jacket with taped seams and adjustable hood."},
    {"name": "Slim-Fit Chinos",       "category": "Clothing",      "price": 54.99,  "description": "Stretch slim-fit chino trousers available in multiple neutral colors."},
    {"name": "Cotton T-Shirt 3-Pack", "category": "Clothing",      "price": 29.99,  "description": "Everyday crewneck cotton tees in classic white, black, and grey."},
    {"name": "Leather Belt",          "category": "Clothing",      "price": 34.99,  "description": "Full-grain leather dress belt with silver-tone pin buckle."},
    {"name": "Hiking Boots",          "category": "Clothing",      "price": 149.99, "description": "Waterproof mid-cut hiking boots with vibram sole and ankle support."},
    {"name": "Puffer Vest",           "category": "Clothing",      "price": 59.99,  "description": "Lightweight down-fill vest with snap front closure and two hand pockets."},
    # Books
    {"name": "Deep Learning Book",    "category": "Books",         "price": 44.99,  "description": "Comprehensive guide to deep learning foundations and modern architectures."},
    {"name": "Python Cookbook",       "category": "Books",         "price": 39.99,  "description": "Practical recipes for writing idiomatic, modern Python code."},
    {"name": "Distributed Systems",   "category": "Books",         "price": 49.99,  "description": "Principles and patterns of building reliable distributed systems at scale."},
    {"name": "Clean Code",            "category": "Books",         "price": 34.99,  "description": "A handbook of agile software craftsmanship principles and best practices."},
    {"name": "The Algorithm Design",  "category": "Books",         "price": 59.99,  "description": "Classic reference covering algorithm design techniques with worked examples."},
    {"name": "Designing Data-Intensive Apps", "category": "Books", "price": 54.99,  "description": "In-depth look at the systems behind modern data-intensive applications."},
    # Home & Garden
    {"name": "French Press Coffee",   "category": "Home & Garden", "price": 29.99,  "description": "Stainless steel French press with double-wall insulation, 34 oz."},
    {"name": "Cast Iron Skillet",     "category": "Home & Garden", "price": 39.99,  "description": "Pre-seasoned 10-inch cast iron skillet suitable for all cooktops."},
    {"name": "Bamboo Cutting Board",  "category": "Home & Garden", "price": 24.99,  "description": "Large reversible bamboo cutting board with juice groove and handle."},
    {"name": "Air Purifier",          "category": "Home & Garden", "price": 119.99, "description": "True HEPA air purifier for rooms up to 500 sq ft, whisper-quiet."},
    {"name": "Succulent Set 6-Pack",  "category": "Home & Garden", "price": 19.99,  "description": "Assorted live succulents in 2-inch pots, easy-care indoor plants."},
    {"name": "Scented Soy Candle",    "category": "Home & Garden", "price": 18.99,  "description": "Hand-poured lavender and vanilla soy wax candle, 50-hour burn time."},
    # Sports
    {"name": "Yoga Mat",              "category": "Sports",        "price": 34.99,  "description": "6mm thick non-slip yoga mat with carry strap, eco-friendly TPE."},
    {"name": "Resistance Bands Set",  "category": "Sports",        "price": 24.99,  "description": "Set of 5 fabric resistance bands with varying tension levels."},
    {"name": "Jump Rope",             "category": "Sports",        "price": 14.99,  "description": "Adjustable speed jump rope with ball-bearing handles and steel cable."},
    {"name": "Water Bottle 32oz",     "category": "Sports",        "price": 29.99,  "description": "Insulated stainless steel water bottle keeps drinks cold 24 hr / hot 12 hr."},
    {"name": "Foam Roller",           "category": "Sports",        "price": 22.99,  "description": "High-density foam roller for muscle recovery and myofascial release."},
    {"name": "Dumbbell Set",          "category": "Sports",        "price": 89.99,  "description": "Adjustable dumbbell set 5-25 lb per hand with compact storage tray."},
]

IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def make_product_image(product: Dict, seed: int = 0) -> np.ndarray:
    """Generate a synthetic product image as a numpy array (H, W, 3) uint8.

    The image is a solid-colored rectangle with the product name overlaid,
    so it's visually distinct per category without requiring real photos.
    """
    rng = random.Random(seed)
    base_color = CATEGORY_COLORS[product["category"]]

    # Add slight per-product color jitter
    color = tuple(
        max(0, min(255, c + rng.randint(-30, 30))) for c in base_color
    )

    img = Image.new("RGB", IMAGE_SIZE, color=color)
    draw = ImageDraw.Draw(img)

    # Draw a centered rounded rectangle as a "product card"
    margin = 20
    draw.rounded_rectangle(
        [margin, margin, IMAGE_SIZE[0] - margin, IMAGE_SIZE[1] - margin],
        radius=15,
        fill=tuple(max(0, c - 40) for c in color),
    )

    # Write product name (word-wrap at ~15 chars)
    words = product["name"].split()
    lines, line = [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) <= 15:
            line.append(word)
        else:
            lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))

    # Try to use a basic font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    total_h = len(lines) * 24
    y = (IMAGE_SIZE[1] - total_h) // 2 - 10
    for line_text in lines:
        bbox = draw.textbbox((0, 0), line_text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((IMAGE_SIZE[0] - w) // 2, y), line_text, fill="white", font=font)
        y += 24

    # Category label at bottom
    cat = product["category"]
    bbox = draw.textbbox((0, 0), cat, font=small_font)
    w = bbox[2] - bbox[0]
    draw.text(((IMAGE_SIZE[0] - w) // 2, IMAGE_SIZE[1] - 35), cat, fill="white", font=small_font)

    return np.array(img)


def image_to_bytes(img_array: np.ndarray, fmt: str = "JPEG") -> bytes:
    """Convert numpy (H,W,3) uint8 array to compressed image bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(raw: bytes) -> np.ndarray:
    """Decode compressed image bytes back to numpy (H,W,3) uint8."""
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic text cleaning: strip, lowercase, collapse whitespace."""
    import re
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def make_training_text(product: Dict) -> str:
    """Combine product fields into a single string for embedding training."""
    return f"{product['name']}. {product['description']} Category: {product['category']}."


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_catalog(
    products: Optional[List[Dict]] = None,
    output_dir: str = "data/raw",
    seed: int = 42,
) -> List[Dict]:
    """Generate the synthetic product catalog and save images to disk.

    Returns a list of records ready to be loaded into a Ray Dataset.
    """
    if products is None:
        products = PRODUCTS

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for i, p in enumerate(products):
        img_array = make_product_image(p, seed=seed + i)
        img_bytes = image_to_bytes(img_array)

        record = {
            "product_id": f"P{i:04d}",
            "name": p["name"],
            "category": p["category"],
            "description": p["description"],
            "price": p["price"],
            "image_bytes": img_bytes,
            "training_text": make_training_text(p),
        }
        records.append(record)

    print(f"Generated {len(records)} products in '{output_dir}'")
    return records


# ---------------------------------------------------------------------------
# Preprocessing helpers (used inside Ray Data map_batches)
# ---------------------------------------------------------------------------

def preprocess_image_batch(batch: Dict) -> Dict:
    """Normalize image bytes to float32 tensor bytes (224x224x3, range [0,1]).

    Suitable as a Ray Data map_batches function.
    """
    processed = []
    for raw in batch["image_bytes"]:
        img = np.array(
            Image.open(io.BytesIO(raw)).convert("RGB").resize(IMAGE_SIZE)
        ).astype(np.float32) / 255.0
        # Store as bytes (float32 little-endian) to keep Parquet schema simple
        processed.append(img.tobytes())
    batch["image_tensor_bytes"] = processed
    return batch


def preprocess_text_batch(batch: Dict) -> Dict:
    """Clean training text. Tokenization happens inside the trainer."""
    batch["text_clean"] = [clean_text(t) for t in batch["training_text"]]
    return batch


def decode_image_tensor(raw: bytes) -> np.ndarray:
    """Inverse of preprocess_image_batch: bytes -> float32 (224,224,3)."""
    return np.frombuffer(raw, dtype=np.float32).reshape(*IMAGE_SIZE, 3)
