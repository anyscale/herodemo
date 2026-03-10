"""
E-Commerce Recommendation System — Streamlit Frontend
======================================================

Lets you upload a product image (or use a demo image) and see
the top-5 recommended products returned by the Ray Serve endpoint.

Run locally (while serve_app is running):
    streamlit run streamlit_app.py

Environment variables:
    SERVE_URL   Base URL of the Ray Serve service  (default: http://localhost:8000)
"""

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVE_URL = os.environ.get("SERVE_URL", "http://localhost:8000")
RECOMMEND_ENDPOINT = f"{SERVE_URL}/recommend"
HEALTH_ENDPOINT = f"{SERVE_URL}/health"

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛍️ E-Commerce Product Recommender")
st.markdown(
    """
    Upload a product image and the system will:
    1. **Caption it** with a vision-language model (BLIP).
    2. **Embed** the caption with a fine-tuned sentence-transformer.
    3. **Return** the top-5 most similar products from the catalog.
    """
)

# ---------------------------------------------------------------------------
# Service health check
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Service Status")
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if resp.ok:
            st.success("✅ Service online")
        else:
            st.error(f"❌ Service returned {resp.status_code}")
    except Exception as e:
        st.warning(f"⚠️ Cannot reach service:\n{e}")

    st.markdown("---")
    st.markdown(f"**Endpoint:** `{RECOMMEND_ENDPOINT}`")
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

st.subheader("Upload a Product Image")

uploaded = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

# Demo image fallback: generate a synthetic headphones-like image
if uploaded is None:
    st.info("No image uploaded — using a demo synthetic image.")
    from utils import make_product_image, PRODUCTS

    demo_product = PRODUCTS[0]  # Wireless Headphones
    demo_arr = make_product_image(demo_product, seed=0)
    demo_pil = Image.fromarray(demo_arr)
    buf = io.BytesIO()
    demo_pil.save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    display_image = demo_pil
else:
    image_bytes = uploaded.read()
    display_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

st.image(display_image, caption="Query image")

# ---------------------------------------------------------------------------
# Call the service
# ---------------------------------------------------------------------------

if st.button("🔍 Find Similar Products", type="primary"):
    with st.spinner("Analysing image and searching catalog …"):
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image_base64": encoded}

        try:
            resp = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the service. "
                "Make sure `serve run serve_app:app` is running."
            )
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                st.code(e.response.text)
            st.stop()

    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------

    st.subheader("📝 Generated Caption")
    st.info(result.get("caption", "(no caption)"))

    st.subheader(f"🏆 Top-{top_k} Recommendations")

    recs = result.get("recommendations", [])[:top_k]
    if not recs:
        st.warning("No recommendations returned.")
    else:
        for rec in recs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{rec['rank']}. {rec['name']}**  \n"
                    f"Category: `{rec['category']}`  \n"
                    f"Product ID: `{rec['product_id']}`"
                )
            with col2:
                sim_pct = int(rec["similarity"] * 100)
                st.metric("Similarity", f"{sim_pct}%")
            st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), "
    "[BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), "
    "and [Sentence-Transformers](https://www.sbert.net/) on Anyscale."
)
