"""Input-only recheck of the morning-path contrast with true opening price."""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import pandas as pd

from .morning_path_continuous import (
    OUTPUT as OLD_OUTPUT,
    ROOT,
    load_inputs,
    select,
)


OUTPUT = ROOT / "morning_path_open_anchor"


def run(output: Path = OUTPUT) -> dict:
    selected, report = select(load_inputs(
        source=ROOT / "morning_open_anchor" / "inputs.parquet",
        opening_anchor="daily_open",
    ))
    old = pd.read_parquet(OLD_OUTPUT / "selections.parquet", columns=[
        "date", "code", "arm",
    ])
    overlap = selected[["date", "code", "arm"]].merge(
        old, on=["date", "code"], suffixes=("_open", "_old"),
        validate="one_to_one",
    )
    report["opening_anchor"] = "independent daily open, known before 14:49"
    report["old_selected_stock_days"] = int(len(old))
    report["new_selected_stock_days"] = int(len(selected))
    report["same_stock_day_selected"] = int(len(overlap))
    report["same_stock_day_and_arm"] = int(
        overlap.arm_open.eq(overlap.arm_old).sum()
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    selected.to_parquet(output / "selections.parquet", index=False,
                        compression="zstd")
    if report["outcome_gate_passed"]:
        sample = selected.copy()
        sample["audit_order"] = [md5(
            ("morning-path-open-v1" + row.date + row.code).encode()
        ).hexdigest() for row in sample.itertuples(index=False)]
        sample = sample.sort_values("audit_order").groupby(
            ["half", "arm"], sort=True).head(40)
        if len(sample) != 320:
            raise ValueError("Incomplete raw-minute input sample")
        sample.drop(columns="audit_order").to_parquet(
            output / "raw_signals.parquet", index=False,
            compression="zstd",
        )
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
