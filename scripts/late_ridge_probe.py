"""Fixed cost-aware ridge ranking with 14:50-safe late-session features.

Train on early 2024, check later 2024, refit on 2024, then check 2025.
The five-session horizon and regularization are fixed before this run.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.market_study import _quality_keys, _quality_symbols, _week_bootstrap
from trade_research.strategy_scan import _stressed_returns


ROOT = Path("data/research")
FEATURES = (
    "return_1450", "return_1450_sq", "position_1450", "log_amount",
    "log_volume_ratio", "return5_prior_adjusted",
    "return20_prior_adjusted", "distance_ma20_adjusted",
    "intraday_range", "overnight_gap", "abs_overnight_gap",
    "return_last30", "abs_return_last30", "volume_share_last30",
    "premium_to_last30_vwap", "log_price", "star", "chinext",
)
HORIZON = 5


def prepare() -> pd.DataFrame:
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")).create_view("s")
    c.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")).create_view("o")
    c.read_parquet(str(ROOT / "intraday_features" / "*.parquet")).create_view("i")
    c.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    c.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    frame = c.execute(f"""
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.amount_1450, s.volume_ratio_est, s.return5_prior_adjusted,
               s.return20_prior_adjusted, s.distance_ma20_adjusted,
               s.high_1450, s.low_1450, s.preclose, s.open_1450,
               s.price_1450, i.return_last30, i.volume_share_last30,
               i.premium_to_last30_vwap,
               o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               NOT EXISTS (SELECT 1 FROM bad_symbols b
                           WHERE b.code = s.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q WHERE q.code = s.code
                     AND q.date >= s.date AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM s JOIN i USING (date, code) JOIN o USING (date, code)
        WHERE o.horizon = {HORIZON}
          AND ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0
          AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
          AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND abs(s.price_1450 - i.price_1450) <= .005
    """).df()
    frame["return_1450_sq"] = frame.return_1450 ** 2
    frame["intraday_range"] = (frame.high_1450 - frame.low_1450) / frame.preclose
    frame["overnight_gap"] = frame.open_1450 / frame.preclose - 1
    frame["abs_overnight_gap"] = frame.overnight_gap.abs()
    frame["abs_return_last30"] = frame.return_last30.abs()
    frame["log_amount"] = np.log(frame.amount_1450)
    frame["log_volume_ratio"] = np.log(frame.volume_ratio_est.clip(lower=.05))
    frame["log_price"] = np.log(frame.price_1450)
    frame["star"] = frame.code.str.startswith("sh.68").astype(int)
    frame["chinext"] = frame.code.str.startswith("sz.30").astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    frame["good"] = (frame.exit_status.eq("filled")
                     & frame.exit_delay_sessions.eq(0)
                     & frame.quality_clean_exit)
    return frame


def fit_and_check(frame: pd.DataFrame, train_end: str,
                  test_first: str, test_last: str) -> dict:
    train = frame.loc[frame.date.le(train_end) & frame.good].copy()
    test = frame.loc[frame.date.between(test_first, test_last)].copy()
    med = train[list(FEATURES)].median()
    scale = (train[list(FEATURES)].quantile(.75)
             - train[list(FEATURES)].quantile(.25)).clip(lower=.01)
    x = ((train[list(FEATURES)] - med) / scale).clip(-5, 5).to_numpy(dtype=float)
    clipped = train.net_return.clip(-.15, .15)
    daily_mean = clipped.groupby(train.date).transform("mean")
    y = (clipped - daily_mean).to_numpy(dtype=float)
    penalty = len(train) * .05
    coefficients = np.linalg.solve(
        x.T @ x + penalty * np.eye(len(FEATURES)), x.T @ y
    )
    test["score"] = (
        ((test[list(FEATURES)] - med) / scale).clip(-5, 5).to_numpy(dtype=float)
        @ coefficients
    )
    test = test.sort_values(["date", "score", "code"],
                            ascending=[True, False, True])
    test["daily_rank"] = test.groupby("date").cumcount() + 1
    chosen = test.loc[test.daily_rank.le(5)].copy()
    valid = chosen.loc[chosen.good].copy()
    dates = sorted(chosen.date.unique())
    per_day = chosen.groupby("date").size().reindex(dates)
    cash = valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / per_day
    valid["stress10"] = _stressed_returns(valid, 10)
    stressed = (valid.groupby("date").stress10.sum().reindex(dates, fill_value=0)
                / per_day)
    universe = frame.loc[frame.date.between(test_first, test_last)].copy()
    universe["cash_return"] = universe.net_return.where(universe.good, 0)
    baseline = universe.groupby("date").cash_return.mean().reindex(dates)
    edge = cash - baseline
    summary = {
        "train_end": train_end, "test_first": test_first,
        "test_last": test_last, "train_rows": len(train),
        "signals": len(chosen), "signal_days": len(dates),
        "entry_fill_rate": float(chosen.entry_status.eq("filled").mean()),
        "clean_ontime_rate": float(len(valid) / len(chosen)),
        "median_trade_net": float(valid.net_return.median()),
        "cash_aware_mean": float(cash.mean()),
        "cash_stress10_mean": float(stressed.mean()),
        "cash_week_ci": _week_bootstrap(
            cash.reset_index(drop=True), pd.Series(dates), 51
        ),
        "same_day_universe_edge": float(edge.mean()),
        "edge_week_ci": _week_bootstrap(
            edge.reset_index(drop=True), pd.Series(dates), 52
        ),
        "coefficients": dict(zip(FEATURES, coefficients.round(6), strict=True)),
    }
    chosen[["date", "code", "daily_rank", "score", "entry_status",
            "entry_price", "shares", "exit_status", "exit_date",
            "exit_delay_sessions", "exit_price", "net_return",
            "quality_clean_exit"]].to_parquet(
                ROOT / f"late_ridge_{test_first[:4]}.parquet", index=False
            )
    connection = duckdb.connect()
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.register("chosen_keys", chosen[["date", "code"]])
    signals = connection.execute("""
        SELECT s.* FROM snapshots s JOIN chosen_keys USING (date, code)
    """).df()
    if len(signals) != len(chosen):
        raise ValueError("A ranked stock-day is absent from the source snapshot")
    signals.to_parquet(ROOT / f"late_ridge_signals_{test_first[:4]}.parquet",
                       index=False, compression="zstd")
    return summary


def main() -> None:
    frame = prepare()
    print({"eligible": len(frame), "days": frame.date.nunique()}, flush=True)
    report = []
    for train_end, first, last in (
        ("2024-06-14", "2024-07-01", "2024-12-17"),
        ("2024-12-17", "2025-01-01", "2025-12-17"),
    ):
        result = fit_and_check(frame, train_end, first, last)
        report.append(result)
        print(result, flush=True)
    (ROOT / "late_ridge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
