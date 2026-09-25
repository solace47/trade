"""Rerun the frozen morning-burst input gate with the true daily open."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .morning_burst import OUTPUT as OLD_OUTPUT, ROOT, _inputs, audit


OUTPUT = ROOT / "morning_open_anchor"


def run(output: Path = OUTPUT) -> dict:
    old_report = json.loads((OLD_OUTPUT / "input_audit.json").read_text(
        encoding="utf-8"))
    inputs, report = audit(_inputs(
        OLD_OUTPUT,
        ROOT / "minute_prefix_1449",
        ROOT / "market_snapshots_ci",
        opening_anchor="daily_open",
    ))
    report["opening_anchor"] = "independent daily open, known before 14:49"
    report["old_proxy_input_count"] = old_report["input_count"]
    report["old_proxy_by_half"] = old_report["by_half"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    inputs.to_parquet(output / "inputs.parquet", index=False,
                      compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
