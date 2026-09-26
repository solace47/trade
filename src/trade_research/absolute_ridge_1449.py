"""Refit the fixed main-board ridge model using only 14:49-labeled inputs."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import FEATURES, ROOT, freeze as freeze_ridge
from .late_to_open_reversal import _board


OUTPUT = ROOT / "absolute_ridge_1449"


def feature_frame(connection: duckdb.DuckDBPyConnection,
                  main_only: bool = True, *, prefix_pattern: str | None = None,
                  date_ranges: tuple[tuple[str, str], ...] | None = None,
                  validation_last: str | None = None) -> pd.DataFrame:
    """Keep legacy feature names while replacing each live value with 14:49 data."""
    if not main_only:
        raise ValueError("The predeclared cutoff sensitivity is main-board only")
    ranges = date_ranges or (("2024-01-01", "2024-12-17"), ("2025-01-01", "2025-12-17"))
    if validation_last is not None and date.fromisoformat(validation_last).year != 2026:
        raise ValueError("The explicit validation cutoff must be in 2026")
    maximum = validation_last or "2025-12-31"
    if any(not "2022-01-01" <= first <= last <= maximum for first, last in ranges):
        raise ValueError("Feature dates must stay within the explicit 2022–2025 research scope")
    conditions = " OR ".join("p.date BETWEEN ? AND ?" for _ in ranges)
    parameters = [value for interval in ranges for value in interval]
    connection.read_parquet(prefix_pattern or str(ROOT / "minute_prefix_1449" / "*" / "*.parquet")
                            ).create_view("prefix_1449")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    missing_amount_eligible = connection.execute("""
        SELECT COUNT(*) FROM prefix_1449 p ANTI JOIN snapshots s USING (date, code)
        WHERE p.date BETWEEN ? AND ?
          AND p.amount_1449 BETWEEN 100000000 AND 1000000000
    """, [min(first for first, _ in ranges), max(last for _, last in ranges)]).fetchone()[0]
    if missing_amount_eligible:
        raise ValueError("A 14:49 amount-eligible stock-day lacks a daily-state join")
    frame = connection.execute(f"""
        SELECT p.date, p.code, p.price_1449, p.amount_1449,
               p.volume_1449, p.high_1449, p.low_1449,
               p.volume_last29, p.amount_last29, p.price_1420,
               s.preclose, s.open_1450, s.volume5_prior,
               s.return5_prior_adjusted, s.return20_prior_adjusted,
               s.ma20_prior_adjusted
        FROM prefix_1449 p JOIN snapshots s USING (date, code)
        WHERE ({conditions})
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
          AND p.amount_1449 BETWEEN 100000000 AND 1000000000
          AND p.price_1449 > 0 AND p.price_1420 > 0
          AND p.volume_1449 > 0 AND p.volume_last29 > 0
          AND s.preclose > 0 AND s.volume5_prior > 0
    """, parameters).df()
    if ((frame.volume_last29 > frame.volume_1449).any()
            or (frame.amount_last29 > frame.amount_1449 + .01).any()):
        raise ValueError("A last-29-minute aggregate exceeds its 14:49 prefix")
    # The old feature names are retained only so the same fixed ridge columns,
    # scaling and penalty can be reused. All intraday values below end at 14:49.
    frame["return_1450"] = frame.price_1449 / frame.preclose - 1
    frame["return_1450_sq"] = frame.return_1450 ** 2
    span = frame.high_1449 - frame.low_1449
    frame["position_1450"] = (frame.price_1449 - frame.low_1449) / span.where(span > 0)
    frame["log_amount"] = np.log(frame.amount_1449)
    volume_ratio = frame.volume_1449 / frame.volume5_prior * (241 / 230)
    frame["log_volume_ratio"] = np.log(volume_ratio.clip(lower=.05))
    frame["distance_ma20_adjusted"] = (
        frame.price_1449 / frame.ma20_prior_adjusted - 1
    )
    frame["intraday_range"] = span / frame.preclose
    frame["overnight_gap"] = frame.open_1450 / frame.preclose - 1
    frame["abs_overnight_gap"] = frame.overnight_gap.abs()
    frame["return_last30"] = frame.price_1449 / frame.price_1420 - 1
    frame["abs_return_last30"] = frame.return_last30.abs()
    frame["volume_share_last30"] = frame.volume_last29 / frame.volume_1449
    frame["premium_to_last30_vwap"] = (
        frame.price_1449 / (frame.amount_last29 / frame.volume_last29) - 1
    )
    frame["log_price"] = np.log(frame.price_1449)
    frame["star"] = frame.code.str.startswith("sh.68").astype(int)
    frame["chinext"] = frame.code.str.startswith("sz.30").astype(int)
    frame["board"] = frame.code.map(_board)
    frame["price_signal"] = frame.price_1449
    frame["amount_signal"] = frame.amount_1449
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    frame = frame.loc[frame.board.eq("main")].copy()
    if frame.empty or frame.duplicated(["date", "code"]).any():
        raise ValueError("Invalid 14:49 main-board ridge feature universe")
    return frame


def freeze(output_dir: Path = OUTPUT) -> dict:
    return freeze_ridge(
        output_dir, main_only=True, feature_builder=feature_frame,
        decision_price_column="price_1449", cutoff_label="1449",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    if args.inputs_only:
        frame = feature_frame(duckdb.connect())
        print({"rows": len(frame), "days": frame.date.nunique(),
               "first": frame.date.min(), "last": frame.date.max()})
    else:
        report = freeze()
        print({key: report[key] for key in (
            "feature_rows", "signals", "controls", "control_fraction",
            "by_half", "outcome_gate_passed",
        )})


if __name__ == "__main__":
    main()
