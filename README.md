# E-Commerce Recommendation System Demo

An end-to-end recommendation system using Ray Data, Ray Train, and Ray Serve on Anyscale.

## Main Walkthrough

The main notebook is in **`notebook.ipynb`**.

## Run as Anyscale Jobs

```bash
# Stage 1: Preprocess product data
anyscale job submit -f setup/job_preprocess.yaml

# Stage 2: Fine-tune embedding model
anyscale job submit -f setup/job_train.yaml
```

## Deploy as a Service

```bash
anyscale service deploy -f setup/service.yaml

# Check status
anyscale service status --name ecomm-recommender-service
```

## Start a Workspace

1. `anyscale workspace_v2 create --name <your-workspace-name> --config-file workspace.yaml`
1. Then search for it in the console: https://console.anyscale.com/workspaces
1. Start the workspace
1. Click on `VSCode`
1. In the terminal clone the repo: `git clone https://github.com/anyscale/herodemo.git`



