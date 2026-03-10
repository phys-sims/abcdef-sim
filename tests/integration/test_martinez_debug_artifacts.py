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
        "martinez_section_iv_phi3_variants.json",
        "martinez_section_iv_phi3_variants.png",
        "treacy_phi3_variant_comparison.json",
        "treacy_phi3_variant_comparison.png",
        "treacy_phi3_per_grating_budget.json",
        "treacy_phi3_per_grating_budget.png",
        "treacy_phi3_sign_audit.json",
        "treacy_phi3_sign_audit.png",
    ):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    phase_payload = json.loads((tmp_path / "martinez_section_iv_phi3_variants.json").read_text())
    treacy_payload = json.loads((tmp_path / "treacy_phi3_variant_comparison.json").read_text())
    budget_payload = json.loads((tmp_path / "treacy_phi3_per_grating_budget.json").read_text())
    sign_payload = json.loads((tmp_path / "treacy_phi3_sign_audit.json").read_text())

    phase_point_keys = set(phase_payload["points"][0].keys())
    treacy_point_keys = set(treacy_payload["points"][0].keys())
    budget_point_keys = set(budget_payload["points"][0].keys())
    sign_stage_keys = set(sign_payload["stage_points"][0].keys())
    sign_variant_keys = set(sign_payload["variant_points"][0].keys())

    assert "variant" in phase_point_keys
    assert "gdd_abs_error_fs2" in phase_point_keys
    assert "tod_abs_error_fs3" in phase_point_keys

    assert "variant" in treacy_point_keys
    assert "raw_abcdef_gdd_rel_error" in treacy_point_keys
    assert "raw_abcdef_tod_rel_error" in treacy_point_keys
    assert "instance_name" in budget_point_keys
    assert "required_residual_gdd_fs2" in budget_point_keys
    assert "return_flip_hypothesis" in sign_stage_keys
    assert "sign_case" in sign_variant_keys
