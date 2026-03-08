from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_treacy_benchmark_artifact_script_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_treacy_benchmark_artifacts.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    for filename in (
        "treacy_radius_convergence.json",
        "treacy_radius_convergence.png",
        "treacy_radius_mirror_heatmap.json",
        "treacy_radius_mirror_heatmap.png",
    ):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
