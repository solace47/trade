"""Split pre-14:50 tail price movement into persistent VWAP and last print.

The input freeze never reads outcomes. The evaluation uses executed 14:52--
14:55 prices, so no decision quote is shared with its return denominator.
"""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _quality_symbols


ROOT = Path("data/research")
OUTPUT = ROOT / "tail_vwap_decomposition"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
REGRESSORS = ("robust_pp", "noise_pp", "early_pp", "day_pp", "prior20_pp",
              "position_1449", "log_price", "log_amount")
SEED = 20260926


def _inputs(candidate_path: Path, anchor_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(candidate_path)).create_view("candidates")
        connection.read_parquet(str(anchor_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        result = connection.execute("""
            SELECT c.*, 100 * (p.vwap_1446_1449 / p.price_1435 - 1)
                   AS robust_pp,
                   100 * (c.price_1449 / p.vwap_1446_1449 - 1)
                   AS noise_pp
            FROM candidates c JOIN prefix p USING (date, code)
            WHERE p.vwap_1446_1449 > 0 AND p.price_1435 > 0
        """).df()
    finally:
        connection.close()
    result["group"] = np.select(
        [result.robust_pp.ge(.2), result.robust_pp.between(-.05, .05)],
        ["rising", "flat"], default="other")
    return result


def _group_coverage(inputs: pd.DataFrame) -> dict:
    counts = inputs.groupby(["half", "date", "board", "group"]).size().unstack(
        "group", fill_value=0)
    for column in ("rising", "flat"):
        if column not in counts:
            counts[column] = 0
    valid = counts.rising.ge(5) & counts.flat.ge(5)
    result = {}
    for half in HALVES:
        part = counts.loc[half]
        selected = valid.loc[half]
        result[half] = {
            "candidates": int(inputs.half.eq(half).sum()),
            "days": int(inputs.loc[inputs.half.eq(half), "date"].nunique()),
            "comparable_date_board_groups": int(selected.sum()),
            "comparable_days": int(part.index.get_level_values("date")[selected].nunique()),
            "rising_inputs": int(part.rising.sum()),
            "flat_inputs": int(part.flat.sum()),
        }
    return result


def _audit_sample(inputs: pd.DataFrame) -> pd.DataFrame:
    selected = inputs.loc[inputs.group.isin(("rising", "flat"))].copy()
    selected["audit_order"] = [
        md5(("tail-vwap-decomp-v1" + row.date + row.code).encode()).hexdigest()
        for row in selected.itertuples(index=False)
    ]
    sample = (selected.sort_values("audit_order")
              .groupby(["half", "group"], sort=True).head(40))
    if len(sample) != 320 or sample.groupby(["half", "group"]).size().ne(40).any():
        raise ValueError("Incomplete frozen raw-minute audit sample")
    return sample.drop(columns="audit_order").sort_values(
        ["half", "group", "date", "code"])


def freeze(candidate_path: Path = ROOT / "execution_drift" / "candidates.parquet",
           anchor_dir: Path = ROOT / "minute_prefix_1449_vwap",
           output_dir: Path = OUTPUT) -> dict:
    source_audit = json.loads((anchor_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not source_audit["input_gate_passed"]:
        raise ValueError("The pre-decision anchor source failed its input audit")
    candidates = pd.read_parquet(candidate_path)
    inputs = _inputs(candidate_path, anchor_dir)
    if (len(inputs) != len(candidates)
            or inputs.duplicated(["date", "code"]).any()
            or not np.isfinite(inputs[list(REGRESSORS)].to_numpy()).all()
            or set(inputs.half) != set(HALVES)):
        raise ValueError("Tail VWAP inputs are incomplete or nonunique")
    groups = _group_coverage(inputs)
    gate = all(row["candidates"] >= 50_000 and row["days"] >= 100
               and row["comparable_date_board_groups"] >= 200
               and row["comparable_days"] >= 80 for row in groups.values())
    sample = _audit_sample(inputs) if gate else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False,
                      compression="zstd")
    if gate:
        sample.to_parquet(output_dir / "raw_signals.parquet", index=False,
                          compression="zstd")
    else:
        (output_dir / "raw_signals.parquet").unlink(missing_ok=True)
    audit = {"periods": list(HALVES), "source_cutoff": "14:49",
             "outcome_gate_passed": gate, "by_half": groups,
             "frozen_raw_sample": len(sample)}
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _outcomes(inputs: pd.DataFrame, outcomes_dir: Path,
              issues_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.register("inputs", inputs)
        connection.register("bad_days", _quality_keys(issues_dir))
        connection.register("bad_symbols", _quality_symbols(issues_dir))
        connection.read_parquet(str(outcomes_dir / "*.parquet")
                                ).create_view("outcomes")
        joined = connection.execute("""
            SELECT i.*, o.horizon, o.entry_status, o.entry_price,
                   o.exit_status, o.exit_price, o.shares,
                   o.target_exit_date, o.exit_date, o.exit_delay_sessions,
                   o.net_return, o.corporate_action_crossed,
                   NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = i.code)
                   AND NOT EXISTS (
                       SELECT 1 FROM bad_days d WHERE d.code = i.code
                         AND d.date >= i.date AND d.date <= o.exit_date
                   ) AS quality_clean
            FROM inputs i JOIN outcomes o USING (date, code)
            WHERE o.horizon IN (1, 5)
              AND o.entry_label = '1452-1455'
              AND o.exit_label = '1452-1455'
        """).df()
    finally:
        connection.close()
    if (len(joined) != len(inputs) * 2
            or joined.duplicated(["date", "code", "horizon"]).any()):
        raise ValueError("Executed outcome join is incomplete or nonunique")
    joined["ready"] = (
        joined.entry_status.eq("filled") & joined.exit_status.eq("filled")
        & joined.exit_delay_sessions.eq(0)
        & joined.quality_clean & joined.net_return.notna()
        & joined.corporate_action_crossed.eq(False)
    )
    return joined


def _net_with_slippage(frame: pd.DataFrame, bps: float) -> np.ndarray:
    """Reprice filled post-2023 sales at a different per-side slip."""
    shares = frame.shares.to_numpy(dtype=float)
    raw_entry = frame.entry_price.to_numpy(dtype=float) / 1.0005
    raw_exit = frame.exit_price.to_numpy(dtype=float) / .9995
    buy = shares * raw_entry * (1 + bps / 10_000)
    sell = shares * raw_exit * (1 - bps / 10_000)
    buy_fees = np.maximum(5, buy * .0003) + buy * .00001
    sell_fees = np.maximum(5, sell * .0003) + sell * .00001 + sell * .0005
    return (sell - sell_fees) / (buy + buy_fees) - 1


def _regression(frame: pd.DataFrame, draws: int = 500) -> dict:
    selected = frame.copy()
    selected["net_pp"] = selected.net_return * 100
    cols = [*REGRESSORS, "net_pp"]
    grouped = selected.groupby(["date", "board"], sort=False)
    selected = selected.loc[grouped.code.transform("size").ge(20)].copy()
    selected[cols] = selected[cols] - selected.groupby(
        ["date", "board"])[cols].transform("mean")
    selected["week"] = pd.to_datetime(selected.date).dt.to_period("W-SUN").astype(str)
    moments = []
    for _, week in selected.groupby("week", sort=True):
        x = week[list(REGRESSORS)].to_numpy(dtype=float)
        y = week.net_pp.to_numpy(dtype=float)
        moments.append((x.T @ x, x.T @ y))
    if len(moments) < 12:
        raise ValueError("Too few weeks for regression")
    matrices = np.stack([a for a, _ in moments])
    vectors = np.stack([b for _, b in moments])
    beta = np.linalg.solve(matrices.sum(axis=0), vectors.sum(axis=0))
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(draws):
        indexes = rng.integers(0, len(moments), len(moments))
        estimate = np.linalg.solve(matrices[indexes].sum(axis=0),
                                   vectors[indexes].sum(axis=0))
        samples.append([estimate[0], estimate[1], estimate[0] - estimate[1]])
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {
        "robust_pp_outcome_pp_per_input_pp": float(beta[0]),
        "noise_pp_outcome_pp_per_input_pp": float(beta[1]),
        "robust_minus_noise": float(beta[0] - beta[1]),
        "week_bootstrap_95": {
            "robust": bounds[:, 0].tolist(),
            "noise": bounds[:, 1].tolist(),
            "difference": bounds[:, 2].tolist(),
        },
        "weeks": len(moments), "rows": len(selected),
    }


def _week_interval(values: pd.Series, dates: pd.Series,
                   draws: int = 500) -> list[float]:
    frame = pd.DataFrame({"value": values.to_numpy(), "date": dates.to_numpy()})
    frame["week"] = pd.to_datetime(frame.date).dt.to_period("W-SUN").astype(str)
    weekly = frame.groupby("week").value.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    indexes = rng.integers(0, len(weekly), size=(draws, len(weekly)))
    sampled = weekly["sum"].to_numpy()[indexes].sum(axis=1) / \
        weekly["count"].to_numpy()[indexes].sum(axis=1)
    return np.quantile(sampled, [.025, .975]).tolist()


def _same_day(frame: pd.DataFrame) -> dict:
    subset = frame.loc[frame.ready & frame.group.isin(("rising", "flat"))].copy()
    metrics = {"net_5bps_pp": subset.net_return.to_numpy(dtype=float) * 100}
    for slip in (10, 15):
        metrics[f"net_{slip}bps_pp"] = _net_with_slippage(subset, slip) * 100
    for name, values in metrics.items():
        subset[name] = values
    grouped = subset.groupby(["date", "board", "group"])
    means = grouped[[*metrics]].mean()
    means["n"] = grouped.size()
    pivot = means.unstack("group")
    pivot = pivot.loc[pivot[("n", "rising")].ge(5)
                      & pivot[("n", "flat")].ge(5)].copy()
    if pivot.empty:
        raise ValueError("No completed same-day comparisons")
    result = {"date_board_groups": len(pivot),
              "days": int(pivot.index.get_level_values("date").nunique())}
    dates = pd.Series(pivot.index.get_level_values("date"), index=pivot.index)
    for name in metrics:
        difference = pivot[(name, "rising")] - pivot[(name, "flat")]
        result[name] = {
            "rising": float(pivot[(name, "rising")].mean()),
            "flat": float(pivot[(name, "flat")].mean()),
            "rising_minus_flat": float(difference.mean()),
            "week_bootstrap_95": _week_interval(difference, dates),
        }
    return result


def evaluate(input_dir: Path = OUTPUT,
             outcomes_dir: Path = ROOT / "market_outcomes_ci",
             issues_dir: Path = ROOT / "market_issues_ci") -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input-only gate failed; outcome archive must stay closed")
    inputs = pd.read_parquet(input_dir / "inputs.parquet")
    joined = _outcomes(inputs, outcomes_dir, issues_dir)
    ready = joined.loc[joined.ready].copy()
    recalculated = _net_with_slippage(ready, 5)
    maximum_error = float(np.max(np.abs(recalculated - ready.net_return)))
    if maximum_error > 1e-10:
        raise ValueError("Five-basis-point repricing disagrees with archive")
    main = joined.loc[joined.horizon.eq(1)]
    by_half = {}
    for half in HALVES:
        period = main.loc[main.half.eq(half)]
        trade = period.loc[period.ready]
        coverage = {}
        for group in ("rising", "flat", "other"):
            part = period.loc[period.group.eq(group)]
            coverage[group] = {
                "inputs": len(part),
                "entry_fill_rate": float(part.entry_status.eq("filled").mean()),
                "on_time_quality_clean_rate": float(part.ready.mean()),
            }
        by_half[half] = {
            "coverage": coverage,
            "regression": _regression(trade),
            "same_day": _same_day(period),
            "t5_same_day": _same_day(joined.loc[
                joined.half.eq(half) & joined.horizon.eq(5)]),
        }
    by_year = {}
    for year in ("2024", "2025"):
        period = main.loc[main.date.str.startswith(year)]
        by_year[year] = {"regression": _regression(period.loc[period.ready]),
                         "same_day": _same_day(period)}
    eligible = all(
        by_half[half]["coverage"][group]["on_time_quality_clean_rate"] >= .90
        for half in HALVES for group in ("rising", "flat")
    )
    direction = (eligible and all(
        by_half[half]["regression"]["robust_minus_noise"] < 0
        and by_half[half]["same_day"]["net_5bps_pp"]["rising_minus_flat"] <= -.20
        for half in HALVES)
        and all(
            by_year[year]["regression"]["week_bootstrap_95"]["difference"][1] < 0
            and by_year[year]["same_day"]["net_5bps_pp"]["week_bootstrap_95"][1] < 0
            for year in ("2024", "2025"))
    )
    result = {
        "years": [2024, 2025], "input_count": len(inputs),
        "five_bps_archive_max_error": maximum_error,
        "by_half": by_half, "by_year": by_year,
        "direction_gate_passed": bool(direction),
        "flat_positive_after_15bps_four_halves": bool(all(
            by_half[half]["same_day"]["net_15bps_pp"]["flat"] > 0
            for half in HALVES)),
        "strategy_formula_released": False,
    }
    (input_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def verify_repriced(input_dir: Path = OUTPUT,
                    outcomes_dir: Path = ROOT / "market_outcomes_ci",
                    issues_dir: Path = ROOT / "market_issues_ci") -> dict:
    sample = pd.read_parquet(input_dir / "raw_signals.parquet")
    repriced = pd.read_parquet(input_dir / "repriced.parquet")
    if len(sample) != 320 or repriced.duplicated(
            ["date", "code", "horizon", "target_notional"]).any():
        raise ValueError("Raw-minute repricing sample or keys are invalid")
    archived = _outcomes(sample, outcomes_dir, issues_dir)
    baseline = repriced.loc[repriced.target_notional.eq(100_000)].merge(
        archived, on=["date", "code", "horizon"],
        suffixes=("_raw", "_archive"), validate="one_to_one")
    if len(baseline) != 640:
        raise ValueError("A frozen 100,000-yuan raw-minute result is absent")
    mismatches = {}
    for column in ("entry_status", "exit_status", "target_exit_date",
                   "exit_date", "corporate_action_crossed"):
        left = baseline[f"{column}_raw"].astype(object).fillna("missing").astype(str)
        right = baseline[f"{column}_archive"].astype(object).fillna("missing").astype(str)
        mismatches[column] = int(left.ne(right).sum())
    for column in ("shares", "exit_delay_sessions", "entry_price", "exit_price",
                   "net_return"):
        left = baseline[f"{column}_raw"].to_numpy(dtype=float)
        right = baseline[f"{column}_archive"].to_numpy(dtype=float)
        mismatches[column] = int((~np.isclose(left, right, atol=1e-8,
                                               rtol=0, equal_nan=True)).sum())
    filled_exit = baseline.exit_status_raw.eq("filled")
    mismatches["quality_clean_exit"] = int((
        baseline.loc[filled_exit, "quality_clean_exit"].astype(bool)
        != baseline.loc[filled_exit, "quality_clean"].astype(bool)
    ).sum())
    if any(mismatches.values()):
        raise ValueError(f"Archived results differ from raw-minute repricing: {mismatches}")
    summary = {"sample_stock_days": len(sample), "repriced_rows": len(repriced),
               "archived_100k_rows_checked": len(baseline),
               "mismatches": mismatches}
    for notional in (20_000, 100_000):
        t1 = repriced.loc[
            repriced.target_notional.eq(notional) & repriced.horizon.eq(1)]
        summary[str(notional)] = {
            "entry_fill_rate": float(t1.entry_status.eq("filled").mean()),
            "on_time_clean_exit_rate": float((
                t1.exit_status.eq("filled") & t1.exit_delay_sessions.eq(0)
                & t1.quality_clean_exit).mean()),
        }
    (input_dir / "raw_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate", "verify-raw"))
    args = parser.parse_args()
    action = {"freeze": freeze, "evaluate": evaluate,
              "verify-raw": verify_repriced}[args.stage]
    print(json.dumps(action(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
