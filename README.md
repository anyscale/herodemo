# E-Commerce Recommendation System Demo

An end-to-end recommendation system using Ray Data, Ray Train, and Ray Serve on Anyscale.

The main notebook is [notebook.ipynb](./notebook.ipynb)

## Start a Workspace from Terminal

1. Install the Anyscale CLI: [Quickstart CLI](https://docs.anyscale.com/reference/quickstart-cli)
1. Clone this repo `git clone https://github.com/anyscale/herodemo.git`
1. `anyscale workspace_v2 create --name <your-workspace-name> --config-file workspace.yaml`
1. `anyscale workspace_v2 start --id <id-output-from-last-step>`
1. Then search for it in the console: https://console.anyscale.com/workspaces

## Deploy as a Service (Optional)

```bash
anyscale service deploy -f setup/service.yaml

# Check status
anyscale service status --name ecomm-recommender-service
```
