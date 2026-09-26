"""Input-only initialization sensitivity of a 2024-onward turnover reference.

The turnover recursion is a statistical proxy, not observed investor holdings.
No future execution price or outcome is read by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .cash_dividend_catalog import ROOT as CATALOG
from .cash_ex_inputs import ROOT as BASE
from .corporate_cash import save_json, sha


ROOT = Path("data/research/turnover_reference")
CALENDAR = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")
BASE_SHA = "9cb00441c701c85ae245f91587806599ac13d151f6b55d76073dacc2f865daf5"
RULE_COMMIT = "a2c86cf"
SUSPENSION_SEMANTICS_COMMIT = "e35421c"


def history_states(history: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """Record end-of-day states; every signal must join a strictly earlier day.

    N is the known, unnormalized price contribution. I is the contribution
    from an initial reference equal to the first 2024 traded close. The three
    hypothetical references are N + k*I, k in {0.5, 1, 2}.
    """
    if history.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate history date/code")
    if not history.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("History outside the permitted years")
    positions = {date: i for i, date in enumerate(calendar)}
    rows = []
    for code, group in history.sort_values(["code", "date"]).groupby("code", sort=False):
        n, initial, mass = 0.0, np.nan, 1.0
        previous_close, previous_float = np.nan, np.nan
        previous_position = None
        first_date = None
        valid = True
        traded_count = price_changes = float_changes = missing_sessions = invalid_rows = 0
        for row in group.itertuples(index=False):
            position = positions.get(row.date)
            if position is None:
                raise ValueError("History day is not in the market calendar")
            if previous_position is not None and position != previous_position + 1:
                missing_sessions += position - previous_position - 1
                valid = False
            previous_position = position
            if pd.notna(row.tradestatus) and row.tradestatus == 0:
                # The provider returns empty volume AND turn on a verified
                # suspension. Preserve those raw nulls; no trade is inferred.
                known_suspension = (pd.notna([row.adjustflag, row.close, row.preclose]).all()
                    and row.adjustflag == 3 and np.isfinite([row.close, row.preclose]).all()
                    and row.close > 0 and abs(row.close - row.preclose) <= .0001
                    and (pd.isna(row.volume) or row.volume == 0)
                    and (pd.isna(row.turn) or row.turn == 0))
                if not known_suspension:
                    invalid_rows += 1
                    valid = False
                continue
            traded_count += 1
            good = (pd.notna([row.tradestatus, row.adjustflag, row.close,
                             row.preclose, row.turn, row.volume]).all()
                    and row.tradestatus == 1 and row.adjustflag == 3
                    and np.isfinite([row.close, row.preclose, row.turn, row.volume]).all()
                    and row.close > 0 and row.preclose > 0 and row.volume > 0
                    and 0 < row.turn <= 100)
            if not good:
                invalid_rows += 1
                valid = False
            if first_date is None:
                first_date = row.date
                initial = float(row.close) if good else np.nan
            elif good and np.isfinite(previous_close):
                scale = row.preclose / previous_close
                n *= scale
                initial *= scale
                price_changes += int(abs(row.preclose - previous_close) > .005)
            if good:
                v = row.turn / 100.0
                current_float = row.volume / v
                if np.isfinite(previous_float):
                    float_changes += int(abs(current_float / previous_float - 1) > .02)
                previous_float = current_float
                n = (1 - v) * n + v * row.close
                initial *= 1 - v
                mass *= 1 - v
            else:
                n = initial = mass = np.nan
            previous_close = row.close if good else np.nan
            rows.append({"code": code, "source_date": row.date, "first_history_date": first_date,
                "history_valid": valid, "history_sessions": traded_count,
                "known_contribution": n if valid else np.nan,
                "initial_contribution": initial if valid else np.nan,
                "initial_mass": mass if valid else np.nan,
                "source_close": row.close, "price_reference_changes": price_changes,
                "implied_float_changes_gt2pct": float_changes,
                "missing_history_sessions": missing_sessions, "invalid_history_rows": invalid_rows})
    return pd.DataFrame(rows)


def attach_references(base: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    if base.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate signal key")
    if not base.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("Signal outside the permitted years")
    out = base.merge(states, left_on=["code", "previous_traded_date"],
                     right_on=["code", "source_date"], how="left", validate="many_to_one")
    if len(out) != len(base):
        raise ValueError("Signal denominator changed")
    source_present = out.source_date.notna()
    if (out.loc[source_present, "source_date"] >= out.loc[source_present, "date"]).any():
        raise ValueError("Reference includes a contemporaneous or future day")
    out["source_association_valid"] = (source_present
        & out.history_sessions.eq(out.prior_sessions)
        & (out.source_close - out.previous_close).abs().le(1e-10))
    out["reference_valid"] = (out.source_association_valid & out.history_valid.fillna(False)
        & np.isfinite(out.price_1449) & out.price_1449.gt(0)
        & np.isfinite(out.preclose) & out.preclose.gt(0)
        & out.initial_mass.lt(1))
    scale = out.preclose / out.source_close
    out["known_contribution_asof"] = (out.known_contribution * scale).where(out.reference_valid)
    out["initial_contribution_asof"] = (out.initial_contribution * scale).where(out.reference_valid)
    out["normalized_reference"] = out.known_contribution_asof / (1 - out.initial_mass)
    out["gain_normalized"] = 1 - out.normalized_reference / out.price_1449
    for label, multiplier in (("half", .5), ("one", 1.0), ("double", 2.0)):
        out[f"reference_seed_{label}"] = (out.known_contribution_asof
                                           + multiplier * out.initial_contribution_asof)
        out[f"gain_seed_{label}"] = 1 - out[f"reference_seed_{label}"] / out.price_1449
    out["gain_scenario_width"] = out.gain_seed_half - out.gain_seed_double
    out["gain_sign_varies"] = (out.reference_valid & out.gain_seed_double.le(0)
                                 & out.gain_seed_half.ge(0)
                                 & out.gain_scenario_width.gt(1e-14))
    return out


def summarize(frame: pd.DataFrame) -> list[dict]:
    summaries = []
    for half, group in frame.groupby("half", sort=True):
        good = group.loc[group.reference_valid]
        row = {"half": half, "all_rows": len(group), "all_dates": int(group.date.nunique()),
               "associated_rows": int(group.source_association_valid.sum()),
               "valid_rows": len(good), "unknown_rows": len(group) - len(good),
               "sign_varies_rows": int(good.gain_sign_varies.sum()),
               "sign_varies_fraction_of_all": float(good.gain_sign_varies.sum() / len(group)),
               "reference_changed_history_rows": int(good.price_reference_changes.gt(0).sum()),
               "float_changed_history_rows": int(good.implied_float_changes_gt2pct.gt(0).sum())}
        for column in ("history_sessions", "initial_mass", "gain_scenario_width"):
            row[column + "_quantiles"] = {str(q): float(good[column].quantile(q))
                if len(good) else None for q in (.1, .5, .9, .99)}
        row["initial_mass_coverage"] = {str(level): {
            "rows": int(good.initial_mass.le(level).sum()),
            "fraction_of_all": float(good.initial_mass.le(level).sum() / len(group))}
            for level in (.01, .05, .1)}
        row["width_le_1pp_rows"] = int(good.gain_scenario_width.le(.01).sum())
        row["width_le_1pp_fraction_of_all"] = row["width_le_1pp_rows"] / len(group)
        summaries.append(row)
    return summaries


def build(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if sha(BASE / "base.parquet") != BASE_SHA:
        raise ValueError("Frozen input population changed")
    base = pd.read_parquet(BASE / "base.parquet")
    bounds = base.groupby("code", as_index=False).agg(history_end=("previous_traded_date", "max"))
    catalog = json.loads((CATALOG / "manifest.json").read_text())
    expected = catalog["daily_sha256"]
    paths = [Path("data/baostock/market_2020_2026/daily") / (code.replace(".", "_") + ".parquet")
             for code in bounds.code]
    source_hashes = {}
    for path in paths:
        digest = sha(path)
        if digest != expected[str(path)]:
            raise ValueError(f"Frozen daily source changed: {path}")
        source_hashes[str(path)] = digest
    manifest = {"rule_commit": RULE_COMMIT, "base_sha256": BASE_SHA,
                "suspension_semantics_commit": SUSPENSION_SEMANTICS_COMMIT,
                "calendar_sha256": sha(CALENDAR), "daily_sha256": source_hashes,
                "first_history_date": "2024-01-01", "years": [2024, 2025],
                "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Refusing to replace a frozen input manifest")
    save_json(output / "manifest.json", manifest)
    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        con.register("bounds", bounds)
        con.read_parquet([str(p) for p in paths]).create_view("daily")
        history = con.execute("""
            SELECT d.date, d.code, d.close, d.preclose, d.turn, d.volume, d.tradestatus, d.adjustflag
            FROM daily d JOIN bounds b USING (code)
            WHERE d.date BETWEEN '2024-01-01' AND '2025-12-31' AND d.date <= b.history_end
            ORDER BY d.code, d.date
        """).df()
        con.read_parquet(str(CALENDAR)).create_view("calendar")
        calendar = con.execute("""
            SELECT calendar_date FROM calendar WHERE is_trading_day = '1'
              AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
            ORDER BY calendar_date
        """).fetchnumpy()["calendar_date"].tolist()
    finally:
        con.close()
    states = history_states(history, calendar)
    frame = attach_references(base, states)
    if len(frame) != 328895:
        raise ValueError("Frozen population row count changed")
    for name, table in (("history_inputs", history), ("history_states", states), ("input_references", frame)):
        table.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    result = {"rule_commit": RULE_COMMIT, "suspension_semantics_commit": SUSPENSION_SEMANTICS_COMMIT,
              "source_stocks": len(paths), "history_rows": len(history),
              "traded_state_rows": len(states), "signal_rows": len(frame), "by_half": summarize(frame),
              "source_association_passed": bool(frame.source_association_valid.all()),
              "output_sha256": {name: sha(output / name) for name in
                  ("history_inputs.parquet", "history_states.parquet", "input_references.parquet")},
              "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False,
              "strategy_gate": "not_defined_input_diagnostic_only",
              "scenario_is_true_cost_bound": False}
    save_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    print(build(parser.parse_args().output))


if __name__ == "__main__":
    main()
