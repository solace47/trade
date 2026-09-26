"""Month-equal calendar comparisons with unknown values and empty slots preserved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .month_turn_1449 import ROOT, RULE_COMMIT
from .reference_gain_accounting import evaluate as account


def slot_mean(values: pd.Series) -> float | None:
    if len(values) > 5:
        raise ValueError("A timing basket has more than five fixed slots")
    return float(values.sum() / 5) if values.notna().all() else None


def month_interval(values: pd.Series) -> list[float] | None:
    """Circular two-month blocks; a missing held position invalidates the interval."""
    if len(values) < 8 or not values.notna().all():
        return None
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(20260926)
    starts = rng.integers(0, len(array), size=(10000, (len(array) + 1) // 2))
    positions = ((starts[:, :, None] + np.arange(2)) % len(array)).reshape(10000, -1)[:, :len(array)]
    return np.quantile(array[positions].mean(axis=1), [.025, .975]).tolist()


def monthly_baskets(rows: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    cells = []
    for month, dates in schedule.groupby("month", sort=True):
        own_date = dates.loc[dates.arm.eq("high"), "date"].item()
        control_date = dates.loc[dates.arm.eq("low"), "date"].item()
        for horizon in (1, 5):
            part = rows.loc[rows.month.eq(month) & rows.horizon.eq(horizon)]
            own, control = part.loc[part.arm.eq("high")], part.loc[part.arm.eq("low")]
            for slip in (5, 15):
                cell = {"month": month, "year": month[:4], "half": dates.half.iloc[0],
                    "own_date": own_date, "control_date": control_date,
                    "horizon": horizon, "slip": slip}
                for name, arm in (("own", own), ("control", control)):
                    cell[name + "_selected"] = len(arm)
                    cell[name + "_cash_slots"] = 5 - len(arm)
                    cell[name + "_bought"] = int(arm.entry_status.eq("filled").sum())
                    cell[name + "_verified_unknown"] = int(arm.known_return5.isna().sum())
                    cell[name + "_clean_ontime"] = int(arm.category.eq("clean_ontime").sum())
                    for kind in ("return", "lower", "upper"):
                        cell[name + "_" + kind] = slot_mean(arm[f"catalog_scenario_{kind}{slip}"])
                for kind, left, right in (("return", "return", "return"),
                    ("lower", "lower", "upper"), ("upper", "upper", "lower")):
                    x, y = cell["own_" + left], cell["control_" + right]
                    cell["difference_" + kind] = None if x is None or y is None else x - y
                cells.append(cell)
    return pd.DataFrame(cells)


def summarize_months(months: pd.DataFrame) -> list[dict]:
    result = []
    groups = list(months.groupby("half")) + list(months.groupby("year")) + [("all", months)]
    for period, block in groups:
        for (horizon, slip), part in block.groupby(["horizon", "slip"]):
            part = part.sort_values("month")
            row = {"period": period, "horizon": int(horizon), "slip": int(slip), "months": len(part)}
            for name in ("own", "control", "difference"):
                for kind in ("return", "lower", "upper"):
                    values = part[name + "_" + kind]
                    row[name + "_" + kind] = float(values.mean()) if values.notna().all() else None
                row[name + "_two_month_interval"] = month_interval(part[name + "_return"]) if "H" not in period else None
                row[name + "_positive_months"] = int(part[name + "_return"].gt(0).sum())
            for name in ("own", "control"):
                row[name + "_bought_fraction_of_slots"] = float(part[name + "_bought"].sum() / (len(part) * 5))
                row[name + "_clean_ontime_fraction_of_slots"] = float(part[name + "_clean_ontime"].sum() / (len(part) * 5))
                row[name + "_verified_unknown"] = int(part[name + "_verified_unknown"].sum())
            result.append(row)
    return result


def evaluate(output: Path = ROOT) -> dict:
    inputs = json.loads((output / "input_report.json").read_text())
    if (sha(output / "signals.parquet") != inputs["signals_sha256"]
            or sha(output / "calendar_schedule.parquet") != inputs["calendar_schedule_sha256"]):
        raise ValueError("The frozen calendar or security list changed")
    accounting = account(output, bootstrap=False)
    signals = pd.read_parquet(output / "signals.parquet")
    schedule = pd.read_parquet(output / "calendar_schedule.parquet")
    rows = pd.read_parquet(output / "catalog_scenario.parquet")
    rows = rows.merge(signals[["date", "code", "month", "daily_rank", "decision_shares"]],
        on=["date", "code"], validate="many_to_one")
    if (len(rows) != len(signals) * 2 or rows.duplicated(["date", "code", "horizon"]).any()
            or not np.array_equal(rows.loc[rows.entry_status.eq("filled"), "shares"],
                rows.loc[rows.entry_status.eq("filled"), "decision_shares"])):
        raise ValueError("The timing execution grid or fixed share quantities changed")
    months = monthly_baskets(rows, schedule)
    if months.month.nunique() != inputs["independent_months"]:
        raise ValueError("A frozen month disappeared from the comparison")
    months.to_parquet(output / "monthly_baskets.parquet", index=False)
    rows.to_parquet(output / "monthly_accounted.parquet", index=False)
    report = {"rule_commit": RULE_COMMIT,
        "interpretation": "month_equal_fixed_slot_scenarios_not_shared_capital_returns_or_same_stock_causal_effect",
        "accounting_scope": "continued" if output.name == "continued" else "original_five_day_exit_delay",
        "summary": summarize_months(months), "execution_rows": len(rows),
        "independent_months": inputs["independent_months"],
        "source_invalid_rows": accounting["source_invalid_rows"],
        "catalogue_action_rows": accounting["catalogue_action_rows"],
        "unresolved_terminal_rows": accounting["unresolved_terminal_rows"],
        "bootstrap": {"unit": "month", "block_length": 2, "circular": True, "replicates": 10000, "seed": 20260926},
        "source_accounting_sha256": sha(output / "catalog_scenario.parquet"),
        "monthly_baskets_sha256": sha(output / "monthly_baskets.parquet"), "holdout_read": False}
    save_json(output / "month_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    report = evaluate(parser.parse_args().output)
    print(json.dumps([r for r in report["summary"] if r["horizon"] == 5 and r["slip"] == 15],
        ensure_ascii=False, indent=2))
