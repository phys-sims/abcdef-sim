from __future__ import annotations

import json
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
        "treacy_spatial_metrics_vs_radius.json",
        "treacy_spatial_metrics_vs_radius.png",
        "treacy_spatial_metrics_vs_radius_mirror.json",
        "treacy_spatial_metrics_vs_radius_mirror.png",
        "treacy_output_plane_spatiospectral.json",
        "treacy_output_plane_spatiospectral.png",
    ):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    radius_payload = json.loads((tmp_path / "treacy_radius_convergence.json").read_text())
    heatmap_payload = json.loads((tmp_path / "treacy_radius_mirror_heatmap.json").read_text())

    radius_point_keys = set(radius_payload["points"][0].keys())
    heatmap_point_keys = set(heatmap_payload["points"][0].keys())

    for keys in (radius_point_keys, heatmap_point_keys):
        assert "raw_abcdef_gdd_fs2" in keys
        assert "raw_abcdef_gdd_rel_error" in keys
        assert "raw_abcdef_tod_fs3" in keys
        assert "raw_abcdef_tod_rel_error" in keys
        assert "full_gdd_fs2" not in keys
        assert "full_gdd_rel_error" not in keys
        assert "full_tod_fs3" not in keys
        assert "full_tod_rel_error" not in keys
        assert "without_phi2_gdd_fs2" not in keys
        assert "without_phi2_tod_fs3" not in keys

    assert heatmap_payload["headline_metric"] == "relative_error_raw_abcdef_vs_analytic"
    assert "raw_abcdef" in heatmap_payload["relative_error_definition"]
