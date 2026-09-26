"""Freeze integer-price-side groups and matched controls using only inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .cash_dividend_catalog import ROOT as CATALOG
from .cash_ex_inputs import ROOT as BASE
from .cash_ex_matching import assess, match_candidates
from .corporate_cash import save_json, sha
from .quote_precision import quote_cents


ROOT = Path("data/research/round_number_1449")
BARS = Path("data/research/noon_recovery_1449/bars")


def price_position(price: float) -> tuple[int, str, int, int]:
    cents = quote_cents(price)
    integer, fraction = divmod(cents, 100)
    if 1 <= fraction <= 19 and 6 <= integer <= 50:
        return cents, "above", integer, fraction
    if 81 <= fraction <= 99 and 6 <= integer + 1 <= 50:
        return cents, "below", integer + 1, 100 - fraction
    return cents, "outside", 0, 0


def assess_round(attempts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    summaries = assess(attempts, pairs)
    for row in summaries:
        part = pairs.loc[pairs.half.eq(row["half"])]
        checks = row["checks"]
        checks.update({"pair_days": row["paired_days"] >= 40, "pairs": row["pairs"] >= 120,
                       "price_1449_balance": row["price_1449_mean_ratio"] is not None
                       and .9 <= row["price_1449_mean_ratio"] <= 1.1})
        row["round_distance_mean_difference_cents"] = (float((part.distance_cents
            - part.control_distance_cents).mean()) if len(part) else None)
        value = row["round_distance_mean_difference_cents"]
        checks["round_distance_balance"] = value is not None and abs(value) <= 3
        for prefix in ("", "control_"):
            row[prefix + "mean_price"] = float(part[prefix + "price_1449"].mean()) if len(part) else None
            row[prefix + "mean_distance_cents"] = float(part[prefix + "distance_cents"].mean()) if len(part) else None
        row["passed"] = all(checks.values())
    return summaries


def freeze(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    base_report = json.loads((BASE / "base_report.json").read_text())
    event_report = json.loads((CATALOG / "primary_supplement_report.json").read_text())
    bar_files = sorted(BARS.glob("part_*.parquet"))
    if not base_report["association_passed"] or not bar_files:
        raise ValueError("Verified 2024-onward base and decision bars are required")
    hashes = {str(BASE / "base.parquet"): base_report["base_sha256"],
              str(CATALOG / "events_augmented.parquet"): event_report["augmented_events_sha256"],
              **{str(path): sha(path) for path in bar_files}}
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError("A verified input changed")
    manifest = {"rule_commit": "794c452", "input_sha256": hashes, "history_start": "2024-01-01",
                "last_input_minute": "1449", "years": [2024, 2025],
                "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen integer-price inputs")
    save_json(manifest_path, manifest)
    base = pd.read_parquet(BASE / "base.parquet")
    events = pd.read_parquet(CATALOG / "events_augmented.parquet")[["code", "dividOperateDate"]].rename(
        columns={"dividOperateDate": "date"})
    pool = base.merge(events.assign(has_event=True), on=["code", "date"], how="left", validate="one_to_one")
    # Broad bounds enclose all permitted integer +/- 19-cent groups. They
    # deliberately include the outside boundaries, before exact-cent recovery.
    pool = pool.loc[~pool.reference_gap & pool.has_event.isna() & pool.price_1449.between(5.8, 50.2)].copy()
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.register("pool", pool)
        connection.read_parquet([str(path) for path in bar_files]).create_view("source_bars")
        joined = connection.execute("""
            SELECT p.*, b.n_1449, b.raw_price_1449, b.valid_1449
            FROM pool p LEFT JOIN source_bars b USING (date, code)
            WHERE p.date BETWEEN '2024-01-01' AND '2025-12-31'
            ORDER BY p.date, p.code
        """).df()
    finally:
        connection.close()
    if len(joined) != len(pool) or joined.duplicated(["date", "code"]).any():
        raise ValueError("Source-bar association multiplied a decision stock day")
    association = []
    for half, group in joined.groupby("half", sort=True):
        unique = group.n_1449.eq(1)
        aligned = (group.raw_price_1449 - group.price_1449).abs().le(.0001)
        association.append({"half": half, "rows": len(group), "unique_bar_rows": int(unique.sum()),
            "price_aligned_rows": int((unique & aligned).sum()), "coverage": float(unique.mean()),
            "valid_last_bar_rows": int(group.valid_1449.fillna(False).sum()),
            "passed": bool(unique.mean() >= .999 and aligned.loc[unique].all())})
    joined = joined.loc[joined.n_1449.eq(1) & joined.valid_1449.fillna(False)].copy()
    positions, invalid = [], []
    for row in joined.itertuples():
        try:
            positions.append(price_position(row.price_1449))
        except ValueError:
            invalid.append({"date": row.date, "code": row.code, "price_1449": row.price_1449})
    if invalid or len(association) != 4 or not all(row["passed"] for row in association):
        report = {"input_gate_passed": False, "association": association, "invalid_quotes": invalid,
                  "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
        save_json(output / "input_report.json", report)
        return report
    for index, column in enumerate(("quote_cents", "side", "round_yuan", "distance_cents")):
        joined[column] = [value[index] for value in positions]
    sides = joined.loc[joined.side.ne("outside")].copy()
    sides.to_parquet(output / "side_inputs.parquet", index=False)
    upper = sides.loc[sides.side.eq("above")].sort_values(["date", "distance_cents", "code"]).copy()
    upper["daily_rank"] = upper.groupby("date", sort=False).cumcount() + 1
    attempts = upper.loc[upper.daily_rank.le(5)].copy()
    controls = sides.loc[sides.side.eq("below")].copy()
    pairs, missed = match_candidates(attempts, controls,
        ratio_limits={"price_1449": 1.25, "amount_1449": 2.0})
    pairs = pairs.merge(attempts[["date", "code", "distance_cents", "round_yuan"]],
                        on=["date", "code"], validate="one_to_one")
    pairs = pairs.merge(controls[["date", "code", "distance_cents", "round_yuan"]].rename(
        columns={"code": "control_code", "distance_cents": "control_distance_cents", "round_yuan": "control_round_yuan"}),
        on=["date", "control_code"], validate="one_to_one")
    if (len(pairs) + len(missed) != len(attempts) or pairs.duplicated(["date", "code"]).any()
            or pairs.duplicated(["date", "control_code"]).any()):
        raise ValueError("Matching lost attempts or reused a control")
    summaries = assess_round(attempts, pairs)
    for name, frame in (("top_signals", attempts), ("input_pairs", pairs), ("unmatched_attempts", missed)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    counts = sides.groupby(["half", "side"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index()
    report = {"rule_commit": manifest["rule_commit"], "association": association,
        "position_counts": counts.to_dict("records"), "by_half": summaries,
        "input_gate_passed": all(row["passed"] for row in summaries),
        "output_sha256": {name: sha(output / name) for name in
            ("side_inputs.parquet", "top_signals.parquet", "input_pairs.parquet", "unmatched_attempts.parquet")},
        "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    save_json(output / "input_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(freeze(args.output))


if __name__ == "__main__":
    main()
