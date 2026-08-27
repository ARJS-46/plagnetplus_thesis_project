#!/usr/bin/env python3
"""Run one or all frozen v1 grid-search configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Dispatch each JSON config to its method-specific training script."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/v1_grid_search"))
    parser.add_argument("--trial", help="Optional short trial ID, for example s10 or w05")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*/*.json"))
    if args.trial:
        configs = [path for path in configs if path.stem.endswith(f"_{args.trial}")]
    if not configs:
        raise SystemExit("No matching grid configuration was found.")

    scripts = {"simsiam": "train_simsiam.py", "swav": "train_swav.py"}
    for config in configs:
        method = config.parent.name
        command = [sys.executable, str(Path(__file__).parent / scripts[method]), "--config", str(config), "--device", args.device]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
