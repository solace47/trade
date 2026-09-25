import json

import pytest

from trade_research.market_beta_overnight import freeze, gross


def test_failed_refreeze_invalidates_previous_beta_result(tmp_path) -> None:
    output = tmp_path / "beta"
    output.mkdir()
    for name in ("inputs.parquet", "input_audit.json", "gross_report.json"):
        (output / name).write_text("old run", encoding="utf-8")
    prefix_audit = tmp_path / "prefix_audit.json"
    prefix_audit.write_text(json.dumps({"input_gate_passed": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="input audit failed"):
        freeze(prefix_audit_file=prefix_audit, output_dir=output)

    assert not any(output.iterdir())
    with pytest.raises(FileNotFoundError):
        gross(input_dir=output)
