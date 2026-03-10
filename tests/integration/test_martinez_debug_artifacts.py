from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_martinez_debug_artifact_script_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_martinez_debug_artifacts.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    for filename in (
        "martinez_section_iv_matrix_audit.json",
        "martinez_section_iv_matrix_audit.png",
        "martinez_section_iv_phase_partition.json",
        "martinez_section_iv_phase_partition.png",
        "treacy_phi_partition_span_study.json",
        "treacy_phi_partition_span_study.png",
    ):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    phase_payload = json.loads((tmp_path / "martinez_section_iv_phase_partition.json").read_text())
    treacy_payload = json.loads((tmp_path / "treacy_phi_partition_span_study.json").read_text())

    phase_point_keys = set(phase_payload["points"][0].keys())
    treacy_point_keys = set(treacy_payload["points"][0].keys())

    assert "variant" in phase_point_keys
    assert "gdd_abs_error_fs2" in phase_point_keys
    assert "tod_abs_error_fs3" in phase_point_keys

    assert "variant" in treacy_point_keys
    assert "raw_abcdef_gdd_rel_error" in treacy_point_keys
    assert "raw_abcdef_tod_rel_error" in treacy_point_keys
