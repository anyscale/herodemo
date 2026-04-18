#!/usr/bin/env python3
"""
Thin CLI around Anyscale image builds and compute-config creation.

Uses the installed `anyscale` CLI (see https://docs.anyscale.com/ for auth).

References:
  - Image build: `anyscale image build`
  - Compute config: https://docs.anyscale.com/reference/compute-config-api
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _require_anyscale_cli() -> str:
    exe = shutil.which("anyscale")
    if not exe:
        sys.exit(
            "The `anyscale` CLI was not found on PATH. Install and authenticate it first."
        )
    return exe


def _run(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print("+", " ".join(cmd), flush=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc


def cmd_image_build(args: argparse.Namespace) -> None:
    exe = _require_anyscale_cli()
    cmd = [
        exe,
        "image",
        "build",
        "--containerfile",
        str(Path(args.containerfile).resolve()),
        "--name",
        args.name,
    ]
    if args.ray_version:
        cmd.extend(["--ray-version", args.ray_version])
    if args.cloud_id:
        cmd.extend(["--cloud-id", args.cloud_id])
    _run(cmd, dry_run=args.dry_run)


def cmd_compute_config_create(args: argparse.Namespace) -> None:
    exe = _require_anyscale_cli()
    cmd = [
        exe,
        "compute-config",
        "create",
        "--name",
        args.name,
        "--config-file",
        str(Path(args.config_file).resolve()),
    ]
    _run(cmd, dry_run=args.dry_run)


_IMAGE_URI_RE = re.compile(
    r"Image built successfully with URI:\s*(\S+)", re.IGNORECASE
)
_COMPUTE_CONFIG_RE = re.compile(
    r"Created compute config:\s*'([^']+)'", re.IGNORECASE
)


def cmd_provision(args: argparse.Namespace) -> None:
    """Build a container image, then create a compute config from YAML."""
    exe = shutil.which("anyscale") or "anyscale"

    build_cmd = [
        exe,
        "image",
        "build",
        "--containerfile",
        str(Path(args.containerfile).resolve()),
        "--name",
        args.image_name,
    ]
    if args.ray_version:
        build_cmd.extend(["--ray-version", args.ray_version])
    if args.cloud_id:
        build_cmd.extend(["--cloud-id", args.cloud_id])

    create_cmd = [
        exe,
        "compute-config",
        "create",
        "--name",
        args.compute_config_name,
        "--config-file",
        str(Path(args.compute_config_file).resolve()),
    ]

    if args.dry_run:
        _run(build_cmd, dry_run=True)
        _run(create_cmd, dry_run=True)
        return

    if exe == "anyscale":
        exe = _require_anyscale_cli()
        build_cmd[0] = exe
        create_cmd[0] = exe

    build_proc = subprocess.run(
        build_cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    build_out = (build_proc.stdout or "") + (build_proc.stderr or "")
    if build_proc.stdout:
        sys.stdout.write(build_proc.stdout)
        sys.stdout.flush()
    if build_proc.stderr:
        sys.stderr.write(build_proc.stderr)
        sys.stderr.flush()
    if build_proc.returncode != 0:
        sys.exit(build_proc.returncode)

    image_m = _IMAGE_URI_RE.search(build_out)
    if not image_m:
        sys.exit(
            "Image build finished but the image URI could not be parsed from CLI output. "
            "Look for a line like: Image built successfully with URI: ..."
        )
    image_uri = image_m.group(1).strip().rstrip(".")

    create_proc = subprocess.run(
        create_cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    create_out = (create_proc.stdout or "") + (create_proc.stderr or "")
    if create_proc.stdout:
        sys.stdout.write(create_proc.stdout)
        sys.stdout.flush()
    if create_proc.stderr:
        sys.stderr.write(create_proc.stderr)
        sys.stderr.flush()
    if create_proc.returncode != 0:
        sys.exit(create_proc.returncode)

    compute_m = _COMPUTE_CONFIG_RE.search(create_out)
    compute_ref = compute_m.group(1) if compute_m else None

    print("\n--- values for workspace / job / service YAML ---", flush=True)
    print(f"image_uri: {image_uri}", flush=True)
    if compute_ref:
        print(
            "# compute_config is usually a name:version string or inline dict; "
            "use the name returned by Anyscale in the UI/CLI.",
            flush=True,
        )
        print(f"# compute_config (registered): {compute_ref}", flush=True)


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of running them.",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wrap Anyscale `image build` and `compute-config create`.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser(
        "image-build",
        help="Run: anyscale image build (Containerfile / Dockerfile).",
    )
    _add_dry_run(p_img)
    p_img.add_argument(
        "-f",
        "--containerfile",
        required=True,
        help="Path to Containerfile or Dockerfile.",
    )
    p_img.add_argument(
        "-n",
        "--name",
        required=True,
        help="Anyscale image name (new version if the name already exists).",
    )
    p_img.add_argument(
        "-r",
        "--ray-version",
        default=None,
        help="Ray version X.Y.Z for this image (optional; Anyscale may default).",
    )
    p_img.add_argument(
        "--cloud-id",
        default=None,
        help="Anyscale Cloud ID (required for Azure control plane only).",
    )
    p_img.set_defaults(func=cmd_image_build)

    p_cc = sub.add_parser(
        "compute-config-create",
        help="Run: anyscale compute-config create (new schema YAML).",
    )
    _add_dry_run(p_cc)
    p_cc.add_argument(
        "-n",
        "--name",
        required=True,
        help="Compute config name (no version suffix).",
    )
    p_cc.add_argument(
        "-f",
        "--config-file",
        required=True,
        help="Path to compute config YAML (new schema).",
    )
    p_cc.set_defaults(func=cmd_compute_config_create)

    p_all = sub.add_parser(
        "provision",
        help="Build image, then create compute config; print image_uri for YAML.",
    )
    _add_dry_run(p_all)
    p_all.add_argument(
        "-f",
        "--containerfile",
        required=True,
        help="Path to Containerfile or Dockerfile.",
    )
    p_all.add_argument(
        "--image-name",
        required=True,
        help="Anyscale image name for `anyscale image build --name`.",
    )
    p_all.add_argument(
        "-r",
        "--ray-version",
        default=None,
        help="Ray version X.Y.Z (optional).",
    )
    p_all.add_argument(
        "--cloud-id",
        default=None,
        help="Anyscale Cloud ID (Azure control plane only).",
    )
    p_all.add_argument(
        "--compute-config-name",
        required=True,
        help="Name for `anyscale compute-config create --name`.",
    )
    p_all.add_argument(
        "--compute-config-file",
        required=True,
        help="Path to compute config YAML for `anyscale compute-config create -f`.",
    )
    p_all.set_defaults(func=cmd_provision)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
