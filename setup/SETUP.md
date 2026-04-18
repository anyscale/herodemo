# Setup

1. Create a container image with pip requirements installed
1. Create a compute config
1. Create a workspace
1. Create template
1. Add data to blob storage

## Anyscale image and compute config (Python helper)

From the repo root, `anyscale_provision.py` wraps the Anyscale CLI for image builds and compute-config creation. It requires the [`anyscale` CLI](https://docs.anyscale.com/reference/quickstart-cli) installed and authenticated on your machine.

Subcommands:

- **`image-build`** — runs `anyscale image build` with `--containerfile`, `--name`, optional `--ray-version`, and optional `--cloud-id` (Azure control plane).
- **`compute-config-create`** — runs `anyscale compute-config create` with `--name` and `--config-file`. The YAML must follow the [compute config schema](https://docs.anyscale.com/reference/compute-config-api) (cluster resources only), not a full `workspace.yaml`. You can copy the `compute_config` block from `workspace.yaml` into its own file as a starting point.
- **`provision`** — runs image build, then compute-config create, and prints `image_uri` plus the registered compute config name for pasting into workspace, job, or service YAML.

Examples:

```bash
python3 anyscale_provision.py image-build \
  -f setup/Dockerfile -n herodemo-image -r 2.55.0

python3 anyscale_provision.py compute-config-create \
  -n herodemo-compute -f your_compute_config.yaml

python3 anyscale_provision.py provision \
  -f setup/Dockerfile \
  --image-name herodemo-image \
  -r 2.55.0 \
  --compute-config-name herodemo-compute \
  --compute-config-file your_compute_config.yaml
```

Use `--dry-run` on any subcommand to print the underlying `anyscale` commands without executing them.

## Development

1. Install the Anyscale CLI: [Quickstart CLI](https://docs.anyscale.com/reference/quickstart-cli)
1. Clone this repo `git clone https://github.com/anyscale/herodemo.git`
1. `anyscale workspace_v2 create --name <your-workspace-name> --config-file workspace.yaml`
1. `anyscale workspace_v2 start --id <id-output-from-last-step>`
1. Then search for it in the console: https://console.anyscale.com/workspaces

