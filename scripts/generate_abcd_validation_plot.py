#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "compare_thick_lens_to_raytracing.py"


def _load_example_module() -> object:
    spec = importlib.util.spec_from_file_location("compare_thick_lens_to_raytracing", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load example module from {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "physics" / "thick_lens_similarity.png",
    )
    args = parser.parse_args()

    module = _load_example_module()
    output_path = module.write_similarity_plot(args.output.resolve())
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
