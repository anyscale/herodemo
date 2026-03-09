Scenario: e-commerce recommendation system (picked because everyone can relate)

Intro and architecture diagram
Use Ray Data to do pre-processing of product data
Use Ray Train with PyTorch to fine-tune embedding model
Ray Serve that takes in an image (image-to-text model) and tells me what product it might be (embeddings model)
Streamlit webpage to test out the service


README (very short, main content in notebook)
Main Notebook
Utils Notebook (hide some complexity)
Compute Config (using declarative config)
Container Image (most stable approach, faster startup)
Job YAML
Service YAML


- should run quickly
- can run on either cpu or gpu
- use very small models