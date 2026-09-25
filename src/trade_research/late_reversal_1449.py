"""Freeze a 14:49-cutoff sensitivity of the late-decline reversal test.

The economic premise and matching thresholds are unchanged from the prior
14:50-label study. This run tests whether its input list survives a cutoff
that is complete by 14:50 under either timestamp convention. It is not a
blind new strategy: 2025 outcomes were viewed in earlier work; 2026 stays out.
Only 14:49 prefix data and historical or contemporaneously known daily
metadata enter selection. Execution and outcomes must be read separately,
after this module's input gate is saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .late_to_open_reversal import select_inputs


ROOT = Path("data/research")
OUTPUT = ROOT / "late_reversal_1449"
MIN_DAYS_PER_HALF = 30


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT) -> dict:
    prefix_audit = json.loads((prefix_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix input gate failed")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")).create_view("daily_metadata")
        connection.execute("""
            CREATE VIEW snapshots AS
            SELECT p.date, p.code, d.return20_prior_adjusted,
                   d.isST, d.listing_age_sessions, d.reference_gap,
                   d.preclose, p.quote_outside_traded_range,
                   p.price_1449 AS price_1450,
                   p.amount_1449 AS amount_1450,
                   p.price_1449 / d.preclose - 1 AS return_1450,
                   (p.price_1449 - p.low_1449)
                       / NULLIF(p.high_1449 - p.low_1449, 0) AS position_1450
            FROM prefix p JOIN daily_metadata d USING (date, code)
        """)
        connection.execute("""
            CREATE VIEW intraday AS
            SELECT date, code, price_1449 AS price_1450,
                   return_last29 AS return_last30
            FROM prefix
        """)
        selected, audit = select_inputs(connection)
        audit["source_cutoff_label"] = "14:49"
        audit["reference_label"] = "14:20"
        audit["source_tail_minutes"] = 29
        audit["matched_return"] = "return_1449"
        audit["minimum_days_per_half"] = MIN_DAYS_PER_HALF
        audit["outcome_gate_passed"] = (
            audit["outcome_gate_passed"]
            and all(item["days"] >= MIN_DAYS_PER_HALF for item in audit["by_half"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
            (output_dir / stale).unlink(missing_ok=True)
        selected = selected.rename(columns={
            "price_1450": "price_1449", "amount_1450": "amount_1449",
            "return_1450": "return_1449", "position_1450": "position_1449",
            "return_last30": "return_last29",
        })
        selected.to_parquet(output_dir / "selections.parquet", index=False,
                            compression="zstd")
        if audit["outcome_gate_passed"]:
            connection.register("selected_keys", selected[["date", "code"]])
            repricing = connection.execute("""
                SELECT s.date, s.code, s.isST, s.reference_gap,
                       s.listing_age_sessions, s.quote_outside_traded_range,
                       s.price_1450 AS price_1449
                FROM selected_keys k JOIN snapshots s USING (date, code)
            """).df()
            if (len(repricing) != len(selected)
                    or repricing.duplicated(["date", "code"]).any()):
                raise ValueError("A selected stock lacks one repricing input")
            repricing.to_parquet(output_dir / "repricing_signals.parquet",
                                 index=False, compression="zstd")
        (output_dir / "input_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return audit
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
