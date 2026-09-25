"""Chronologically train stock-level margin-interest slopes, then select.

This module fits only on 2024 outcomes that finished before each test starts.
It does not inspect the 2024-H2 or 2025 test outcomes during selection.
The original frozen plan remains in Git history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .margin_dtc_study import MAX_DISTANCE, _distance
from .market_study import _quality_keys, _quality_symbols


WINDOWS = (
    ("2024_H2", "2024-01-02", "2024-06-14", "2024-07-01",
     "2024-12-17", 80),
    ("2025", "2024-01-02", "2024-12-17", "2025-01-02",
     "2025-12-17", 160),
)
CAPACITY = 5
COOLDOWN_SESSIONS = 10
MIN_INTEREST_Z = .5
MIN_HAC_T = 1.645
TREATED = "trained_positive_margin"
CONTROL = "same_day_nontrigger_margin"


def _hac_fit(group: pd.DataFrame, minimum: int) -> dict | None:
    """OLS with Bartlett-weighted Newey-West lag-five covariance."""
    if len(group) < minimum:
        return None
    group = group.sort_values("date")
    interest = group.interest.to_numpy(dtype=float)
    mean = float(interest.mean())
    std = float(interest.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        return None
    x = np.column_stack((
        np.ones(len(group)), (interest - mean) / std,
        group.return20_prior_adjusted.to_numpy(dtype=float),
        group.return_1450.to_numpy(dtype=float),
    ))
    y = group.relative_net.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) != 4 or np.linalg.cond(xtx) > 1e8:
        return None
    beta = np.linalg.solve(xtx, x.T @ y)
    residual = y - x @ beta
    xu = x * residual[:, None]
    meat = xu.T @ xu
    for lag in range(1, 6):
        cross = xu[lag:].T @ xu[:-lag]
        meat += (1 - lag / 6) * (cross + cross.T)
    inverse = np.linalg.inv(xtx)
    covariance = inverse @ meat @ inverse * len(group) / (len(group) - 4)
    variance = float(covariance[1, 1])
    if variance <= 0 or not np.isfinite(variance):
        return None
    return {"n": len(group), "interest_mean": mean, "interest_std": std,
            "beta": float(beta[1]), "hac_se": float(np.sqrt(variance)),
            "hac_t": float(beta[1] / np.sqrt(variance))}


def _prepare_inputs(universe_path: Path, margin_path: Path) -> pd.DataFrame:
    universe = pd.read_parquet(universe_path)
    margin = pd.read_parquet(margin_path,
                             columns=["trade_date", "code", "balance_yuan"])
    frame = universe.merge(margin, on=["trade_date", "code"],
                           validate="many_to_one")
    if len(frame) != len(universe):
        raise ValueError("A margin universe stock-day lost its financing balance")
    frame = frame.loc[
        frame.avg20_amount.ge(100_000_000)
        & frame.amount_1450.ge(100_000_000)
        & frame.float_mv.gt(0)
    ].copy()
    frame["interest"] = frame.balance_yuan / frame.float_mv
    if not np.isfinite(frame.interest).all() or frame.interest.lt(0).any():
        raise ValueError("Malformed stock-level financing interest")
    return frame


def _training_rows(inputs: pd.DataFrame, outcome_dir: Path,
                   issues_dir: Path, train_start: str, train_end: str,
                   test_start: str) -> pd.DataFrame:
    sample = inputs.loc[inputs.date.between(train_start, train_end)]
    if sample.empty or not sample.date.lt(test_start).all():
        raise ValueError("No chronological margin training inputs")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("u", sample)
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    rows = c.execute("""
        SELECT u.date, u.code, u.board, u.size_bucket,
               u.interest, u.return20_prior_adjusted, u.return_1450,
               o.net_return, o.exit_date
        FROM u JOIN o USING(date, code)
        WHERE o.horizon = 5 AND o.entry_status = 'filled'
          AND o.exit_status = 'filled' AND o.exit_date < ?
          AND NOT EXISTS (
              SELECT 1 FROM bad_symbols b WHERE b.code = u.code
          )
          AND NOT EXISTS (
              SELECT 1 FROM bad_days q
              WHERE q.code = u.code AND q.date >= u.date
                AND q.date <= o.exit_date
          )
    """, [test_start]).df()
    if rows.empty or not rows.exit_date.lt(test_start).all():
        raise ValueError("Training uses unfinished or unavailable exits")
    benchmark = rows.groupby(["date", "board", "size_bucket"])[
        "net_return"].transform("mean")
    rows["relative_net"] = rows.net_return - benchmark
    return rows


def _match_one(signal: pd.Series, choices: pd.DataFrame) -> tuple[pd.Series, float] | None:
    eligible = choices.loc[
        (choices.avg20_amount / signal.avg20_amount).between(.5, 2)
        & (choices.float_mv / signal.float_mv).between(.5, 2)
        & (choices.return20_prior_adjusted
           - signal.return20_prior_adjusted).abs().le(.15)
        & (choices.return_1450 - signal.return_1450).abs().le(.05)
    ]
    if eligible.empty:
        return None
    distances = _distance(eligible, signal)
    nearest = int(np.argmin(distances.to_numpy()))
    best = float(distances.iloc[nearest])
    if not np.isfinite(best) or best > MAX_DISTANCE:
        return None
    return eligible.iloc[nearest], best


def _select_window(inputs: pd.DataFrame, models: pd.DataFrame,
                   test_start: str, test_end: str,
                   session_index: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    test = inputs.loc[inputs.date.between(test_start, test_end)].merge(
        models, on="code", how="left", validate="many_to_one")
    test["interest_z"] = (test.interest - test.interest_mean) / test.interest_std
    test["positive_model"] = test.hac_t.gt(MIN_HAC_T) & test.beta.gt(0)
    test["trigger"] = test.positive_model & test.interest_z.ge(MIN_INTEREST_Z)
    test["score"] = test.beta * test.interest_z
    high_rows = []
    low_rows = []
    unmatched = 0
    capacity_selected = 0
    last_kept: dict[str, int] = {}
    for day, daily in test.groupby("date", sort=True):
        index = session_index[day]
        candidates = daily.loc[daily.trigger].sort_values(
            ["score", "code"], ascending=[False, True]
        )
        candidates = candidates.loc[
            candidates.code.map(lambda code: index - last_kept.get(code, -1000)
                                > COOLDOWN_SESSIONS)
        ].head(CAPACITY)
        capacity_selected += len(candidates)
        controls = daily.loc[~daily.trigger].copy()
        for _, signal in candidates.iterrows():
            same = controls.loc[
                controls.board.eq(signal.board)
                & controls.size_bucket.eq(signal.size_bucket)
            ]
            match = _match_one(signal, same)
            if match is None:
                unmatched += 1
                continue
            reference, distance = match
            item = signal.copy()
            item["pair_code"] = signal.code
            item["match_distance"] = distance
            item["candidate"] = TREATED
            high_rows.append(item)
            other = reference.copy()
            other["pair_code"] = signal.code
            other["match_distance"] = distance
            other["candidate"] = CONTROL
            low_rows.append(other)
            controls = controls.loc[controls.code.ne(reference.code)]
            last_kept[signal.code] = index
    if not high_rows:
        raise ValueError("No chronologically trained margin signal was matched")
    selected = pd.concat([pd.DataFrame(high_rows), pd.DataFrame(low_rows)],
                         ignore_index=True)
    if not selected.date.gt(selected.trade_date).all():
        raise ValueError("Margin classifier used same-day financing balance")
    report = {"test_start": test_start, "test_end": test_end,
              "test_stock_days": len(test),
              "positive_model_symbols": int(models.hac_t.gt(MIN_HAC_T).sum()),
              "trigger_stock_days": int(test.trigger.sum()),
              "capacity_selected": capacity_selected,
              "matched_pairs": len(high_rows), "unmatched": unmatched,
              "matched_days": len({row["date"] for row in high_rows}),
              "unique_high_symbols": len({row["code"] for row in high_rows})}
    return selected, report


def fit_and_select(universe_path: Path, margin_path: Path,
                   outcome_dir: Path, issues_dir: Path, calendar: Path,
                   selected_path: Path, models_path: Path,
                   report_path: Path) -> dict:
    inputs = _prepare_inputs(universe_path, margin_path)
    days = trading_dates(calendar, "2024-01-01", "2025-12-31")
    session_index = {day: index for index, day in enumerate(days)}
    all_selected = []
    all_models = []
    report = {"method": "fixed chronological per-stock HAC(5) training",
              "input_stock_days": len(inputs), "windows": {}}
    for name, train_start, train_end, test_start, test_end, minimum in WINDOWS:
        training = _training_rows(inputs, outcome_dir, issues_dir,
                                  train_start, train_end, test_start)
        fitted = []
        for code, group in training.groupby("code", sort=True):
            model = _hac_fit(group, minimum)
            if model is not None:
                fitted.append({"code": code, **model})
        models = pd.DataFrame(fitted)
        if models.empty:
            raise ValueError(f"No {name} stock-level margin models could be fitted")
        models["window"] = name
        selected, selection = _select_window(inputs, models,
                                             test_start, test_end,
                                             session_index)
        selected["window"] = name
        all_selected.append(selected)
        all_models.append(models)
        report["windows"][name] = {
            "train_start": train_start, "train_end": train_end,
            "training_rows": len(training),
            "fitted_symbols": len(models),
            "model_latest_exit": str(training.exit_date.max()),
            **selection,
        }
    final = pd.concat(all_selected, ignore_index=True)
    pair_counts = final.groupby(["date", "pair_code"]).candidate.nunique()
    if (not pair_counts.eq(2).all() or len(final) != 2 * len(pair_counts)
            or final.duplicated(["date", "code", "candidate"]).any()):
        raise ValueError("Incomplete or duplicated trained-margin pairs")
    if not final.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Trained-margin test escaped the 2024-2025 window")
    for path in (selected_path, models_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(selected_path, index=False, compression="zstd")
    pd.concat(all_models, ignore_index=True).to_parquet(
        models_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/margin_trained_selected.parquet"))
    parser.add_argument("--models", type=Path,
                        default=Path("data/research/margin_trained_models.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/margin_trained_selection.json"))
    args = parser.parse_args()
    report = fit_and_select(args.universe, args.margin, args.outcomes,
                            args.issues, args.calendar, args.selected,
                            args.models, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
