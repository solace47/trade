"""Audit the market-wide 14:49 quote to 14:52--14:55 fill-price drift.

Run ``freeze`` before ``evaluate``. The first stage reads only prefix and
same-day pre-decision snapshots; its candidate and raw-minute audit sample are
persisted before the execution-price archive is opened.
"""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .hf_outcomes import Assumptions, _fill, _order_shares


ROOT = Path("data/research")
OUTPUT = ROOT / "execution_drift"
FEATURES = ("tail_pp", "early_pp", "day_pp", "prior20_pp",
            "position_1449", "log_price", "log_amount")
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
ENTRY_LABEL = "1452-1455"
SEED = 20260926


def _half(date: pd.Series) -> pd.Series:
    return date.str[:4] + "H" + np.where(date.str[5:7].astype(int) <= 6, "1", "2")


def _input_connection(prefix_dir: Path, snapshot_dir: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                            ).create_view("prefix")
    connection.from_parquet(str(snapshot_dir / "*.parquet")
                            ).create_view("snapshots")
    return connection


def _candidates(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute("""
        SELECT p.date, p.code,
               CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                    THEN 'main'
                    WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                    WHEN p.code LIKE 'sh.68%' THEN 'star'
                    ELSE NULL END AS board,
               p.price_1449, p.amount_1449, s.preclose,
               s.isST, s.tradestatus, s.reference_gap,
               s.listing_age_sessions, p.quote_outside_traded_range,
               100 * p.return_last14 AS tail_pp,
               100 * (p.price_1435 / p.price_1420 - 1) AS early_pp,
               100 * (p.price_1449 / s.preclose - 1) AS day_pp,
               100 * s.return20_prior_adjusted AS prior20_pp,
               (p.price_1449 - p.low_1449)
                 / NULLIF(p.high_1449 - p.low_1449, 0) AS position_1449,
               LN(p.price_1449) AS log_price,
               LN(p.amount_1449) AS log_amount
        FROM prefix p JOIN snapshots s USING (date, code)
        WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
          AND s.tradestatus = 1 AND s.isST = 0
          AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
          AND NOT p.quote_outside_traded_range
          AND s.preclose > 0 AND p.price_1449 >= 5
          AND p.amount_1449 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND p.return_last29 BETWEEN -.01 AND .01
          AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
          AND (p.price_1449 - p.low_1449)
              / NULLIF(p.high_1449 - p.low_1449, 0) BETWEEN 0 AND 1
          AND isfinite(p.return_last14)
          AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
            OR p.code LIKE 'sz.30%' OR p.code LIKE 'sh.68%')
    """).df()


def _terciles(frame: pd.DataFrame) -> pd.DataFrame:
    """Define within-half input terciles with deterministic tie order."""
    ranked = frame.sort_values(["half", "tail_pp", "date", "code"]).copy()
    offset = ranked.groupby("half").cumcount()
    size = ranked.groupby("half").code.transform("size")
    ranked["input_tercile"] = np.minimum(2, (3 * offset // size)).astype(int)
    return ranked


def _raw_sample(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = _terciles(frame)
    ranked["audit_order"] = [md5(("execution-drift-v1" + row.date + row.code)
                                 .encode()).hexdigest()
                             for row in ranked.itertuples(index=False)]
    sampled = []
    for (_, tercile), group in ranked.groupby(["half", "input_tercile"]):
        count = (20 if tercile == 1 else 30)
        if len(group) < count:
            raise ValueError("Insufficient input-only raw-minute audit candidates")
        sampled.append(group.sort_values("audit_order").head(count))
    return pd.concat(sampled, ignore_index=True).sort_values(
        ["half", "input_tercile", "audit_order"]
    ).drop(columns="audit_order")


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT) -> dict:
    prefix_audit = json.loads((prefix_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix gate failed")
    connection = _input_connection(prefix_dir, snapshot_dir)
    try:
        candidates = _candidates(connection)
    finally:
        connection.close()
    candidates["half"] = _half(candidates.date)
    if candidates.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate candidate stock-day")
    if candidates[list(FEATURES)].isna().any().any():
        raise ValueError("Missing input feature")
    if set(candidates.half) != set(HALVES):
        raise ValueError("Incomplete study periods")
    by_half = candidates.groupby("half").agg(
        candidates=("code", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    gate = all(row["candidates"] >= 50_000 and row["days"] >= 100
               for row in by_half)
    sample = _raw_sample(candidates) if gate else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output_dir / "candidates.parquet", index=False,
                          compression="zstd")
    if gate:
        sample.to_parquet(output_dir / "raw_sample.parquet", index=False,
                          compression="zstd")
    else:
        (output_dir / "raw_sample.parquet").unlink(missing_ok=True)
    audit = {"cutoff": "14:49", "years": [2024, 2025],
             "input_gate_passed": gate, "unique_keys": True,
             "by_half": by_half, "raw_sample_rows": len(sample)}
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _joined_candidates(candidates: pd.DataFrame, outcomes_dir: Path
                       ) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.register("candidates", candidates)
        connection.from_parquet(str(outcomes_dir / "*.parquet")
                                ).create_view("outcomes")
        joined = connection.execute("""
            SELECT c.*, o.entry_status, o.entry_price,
                   o.horizon AS joined_horizon
            FROM candidates c LEFT JOIN outcomes o
              ON o.date = c.date AND o.code = c.code
             AND o.horizon = 1 AND o.entry_label = '1452-1455'
             AND o.exit_label = '1452-1455'
        """).df()
    finally:
        connection.close()
    if len(joined) != len(candidates) or joined.duplicated(["date", "code"]).any():
        raise ValueError("Execution archive join is not one-to-one")
    return joined


def _moments(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = frame.copy()
    columns = [*FEATURES, "drift_bps"]
    group = data.groupby(["date", "board"], sort=False)
    data = data.loc[group.code.transform("size").ge(20)].copy()
    centered = data[columns] - data.groupby(["date", "board"])[columns].transform("mean")
    data[columns] = centered
    data["week"] = pd.to_datetime(data.date).dt.to_period("W-SUN").astype(str)
    moments = {}
    for week, rows in data.groupby("week", sort=True):
        x = rows[list(FEATURES)].to_numpy(dtype=float)
        y = rows.drift_bps.to_numpy(dtype=float)
        moments[week] = (x.T @ x, x.T @ y)
    return moments


def _slope_and_interval(frame: pd.DataFrame, draws: int = 500) -> dict:
    moments = _moments(frame)
    matrices = np.stack([item[0] for item in moments.values()])
    vectors = np.stack([item[1] for item in moments.values()])
    if len(matrices) < 12:
        raise ValueError("Too few observed weeks for interval")
    beta = np.linalg.solve(matrices.sum(axis=0), vectors.sum(axis=0))
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(draws):
        choices = rng.integers(0, len(matrices), len(matrices))
        estimates.append(float(np.linalg.solve(
            matrices[choices].sum(axis=0), vectors[choices].sum(axis=0)
        )[0]))
    return {"tail_bps_per_pp": float(beta[0]),
            "week_bootstrap_95": [float(value) for value in
                                  np.quantile(estimates, [.025, .975])],
            "weeks": len(matrices)}


def _tercile_difference(frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    grouped = frame.groupby(["date", "board"], sort=False)
    frame = frame.loc[grouped.code.transform("size").ge(20)].copy()
    frame["rank_pct"] = frame.groupby(["date", "board"]).tail_pp.rank(
        method="average", pct=True)
    frame["side"] = np.select(
        [frame.rank_pct.le(1 / 3), frame.rank_pct.ge(2 / 3)],
        ["low", "high"], default="middle")
    means = frame.loc[frame.side.ne("middle")].groupby(
        ["date", "board", "side"]
    ).agg(y=("drift_bps", "mean"), x=("tail_pp", "mean"))
    pairs = means.unstack("side").dropna()
    if pairs.empty:
        raise ValueError("No within-day board tercile comparisons")
    return {"date_board_groups": len(pairs),
            "days": pairs.index.get_level_values("date").nunique(),
            "drift_high_minus_low_bps": float((pairs[("y", "high")]
                                                - pairs[("y", "low")]).mean()),
            "tail_high_minus_low_pp": float((pairs[("x", "high")]
                                              - pairs[("x", "low")]).mean())}


def evaluate(input_dir: Path = OUTPUT,
             outcomes_dir: Path = ROOT / "market_outcomes_ci") -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["input_gate_passed"]:
        raise ValueError("Input gate failed; do not read execution outcomes")
    candidates = pd.read_parquet(input_dir / "candidates.parquet")
    joined = _joined_candidates(candidates, outcomes_dir)
    joined["has_archive_row"] = joined.joined_horizon.notna()
    archive_coverage = joined.groupby("half").has_archive_row.mean().to_dict()
    if min(archive_coverage.values()) < .999:
        raise ValueError("Execution archive coverage below preregistered gate")
    joined["filled"] = joined.entry_status.eq("filled") & joined.entry_price.notna()
    filled = joined.loc[joined.filled].copy()
    filled["raw_vwap"] = filled.entry_price / 1.0005
    filled["drift_bps"] = 10_000 * (filled.raw_vwap / filled.price_1449 - 1)
    if not np.isfinite(filled.drift_bps).all():
        raise ValueError("Nonfinite execution drift")
    by_half = {}
    for half in HALVES:
        total = joined.loc[joined.half.eq(half)]
        trades = filled.loc[filled.half.eq(half)]
        by_half[half] = {
            "candidates": len(total), "archive_coverage": archive_coverage[half],
            "buy_fill_rate": float(total.filled.mean()),
            "fill_status": total.entry_status.fillna("missing").value_counts().to_dict(),
            "mean_raw_drift_bps": float(trades.drift_bps.mean()),
            "median_raw_drift_bps": float(trades.drift_bps.median()),
            "entry_cost_bps_by_slippage": {
                str(slip): float((10_000 * (
                    trades.raw_vwap / trades.price_1449 * (1 + slip / 10_000)
                    - 1)).mean()) for slip in (5, 10, 15)
            },
            "regression": _slope_and_interval(trades),
            "same_day_terciles": _tercile_difference(trades),
        }
    by_year = {year: _slope_and_interval(filled.loc[
        filled.date.str.startswith(year)]) for year in ("2024", "2025")}
    pooled = _slope_and_interval(filled)
    associated = (
        all(row["buy_fill_rate"] >= .90 for row in by_half.values())
        and all(row["regression"]["tail_bps_per_pp"] > 0
                and row["same_day_terciles"]["drift_high_minus_low_bps"] > 0
                for row in by_half.values())
        and all(row["week_bootstrap_95"][0] > 0 for row in by_year.values())
        and pooled["tail_bps_per_pp"] >= 4
    )
    result = {"metric": "14:52-14:55 raw VWAP versus 14:49 price",
              "years": [2024, 2025], "by_half": by_half,
              "by_year": by_year, "pooled": pooled,
              "preregistered_execution_association_passed": bool(associated),
              "strategy_formula_released": False}
    (input_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def anchor_sensitivity(input_dir: Path = OUTPUT,
                       anchor_dir: Path = ROOT / "minute_prefix_1449_vwap",
                       outcomes_dir: Path = ROOT / "market_outcomes_ci") -> dict:
    """Exploratory alternate decision quote, designed after primary results."""
    anchor_audit = json.loads((anchor_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not anchor_audit["input_gate_passed"]:
        raise ValueError("Alternative 14:49 prefix audit failed")
    candidates = pd.read_parquet(input_dir / "candidates.parquet")
    connection = duckdb.connect()
    try:
        connection.register("candidates", candidates)
        connection.from_parquet(str(anchor_dir / "*" / "part_*.parquet")
                                ).create_view("anchor_prefix")
        with_anchor = connection.execute("""
            SELECT c.*, p.vwap_1446_1449 AS pre_vwap
            FROM candidates c LEFT JOIN anchor_prefix p USING (date, code)
        """).df()
    finally:
        connection.close()
    if len(with_anchor) != len(candidates) or with_anchor.duplicated(["date", "code"]).any():
        raise ValueError("Pre-decision quote anchor join is not one-to-one")
    anchor_coverage = with_anchor.pre_vwap.gt(0).groupby(with_anchor.half).mean().to_dict()
    if min(anchor_coverage.values()) < .999:
        raise ValueError("Alternative quote anchor coverage below 99.9%")
    joined = _joined_candidates(with_anchor, outcomes_dir)
    joined["filled"] = joined.entry_status.eq("filled") & joined.entry_price.notna()
    filled = joined.loc[joined.filled & joined.pre_vwap.gt(0)].copy()
    filled["raw_vwap"] = filled.entry_price / 1.0005
    filled["drift_bps"] = 10_000 * (filled.raw_vwap / filled.pre_vwap - 1)
    filled["anchor_minus_last_bps"] = 10_000 * (
        filled.pre_vwap / filled.price_1449 - 1)
    by_half = {}
    for half in HALVES:
        trades = filled.loc[filled.half.eq(half)]
        by_half[half] = {
            "filled_rows": len(trades), "anchor_coverage": anchor_coverage[half],
            "mean_drift_bps": float(trades.drift_bps.mean()),
            "mean_anchor_minus_last_bps": float(trades.anchor_minus_last_bps.mean()),
            "regression": _slope_and_interval(trades),
            "same_day_terciles": _tercile_difference(trades),
        }
    result = {
        "exploratory_after_primary_result": True,
        "anchor": "14:46-14:49 four-bar VWAP, known by decision",
        "feature": "original 14:35-14:49 last-trade return",
        "years": [2024, 2025], "by_half": by_half,
        "by_year": {year: _slope_and_interval(filled.loc[
            filled.date.str.startswith(year)]) for year in ("2024", "2025")},
    }
    (input_dir / "anchor_sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def verify_raw_sample(input_dir: Path = OUTPUT,
                      minute_root: Path = Path("data/hf/pilot/data/stock_1m"),
                      outcomes_dir: Path = ROOT / "market_outcomes_ci",
                      anchor_dir: Path = ROOT / "minute_prefix_1449_vwap") -> dict:
    sample = pd.read_parquet(input_dir / "raw_sample.parquet")
    if len(sample) != 320 or sample.duplicated(["date", "code"]).any():
        raise ValueError("Frozen raw-minute sample is invalid")
    joined = _joined_candidates(sample, outcomes_dir)
    anchor_lookup = {}
    if (anchor_dir / "input_audit.json").exists():
        connection = duckdb.connect()
        try:
            connection.register("sample_keys", sample[["date", "code"]])
            connection.from_parquet(str(anchor_dir / "*" / "part_*.parquet")
                                    ).create_view("anchor_prefix")
            anchors = connection.execute("""
                SELECT k.date, k.code, p.vwap_1446_1449
                FROM sample_keys k JOIN anchor_prefix p USING (date, code)
            """).df()
        finally:
            connection.close()
        if len(anchors) != len(sample) or anchors.duplicated(["date", "code"]).any():
            raise ValueError("Frozen raw sample lacks unique alternate quote anchor")
        anchor_lookup = {(row.date, row.code): row.vwap_1446_1449
                         for row in anchors.itertuples(index=False)}
    mismatches = []
    anchor_mismatches = []
    low_size_fills = 0
    for row in joined.itertuples(index=False):
        exchange, symbol = row.code.split(".")
        path = minute_root / exchange.upper() / f"{symbol}.parquet"
        start = pd.Timestamp(row.date)
        minute = pd.read_parquet(path, columns=["timestamp", "volume", "turnover"],
                                 filters=[("timestamp", ">=", start),
                                          ("timestamp", "<", start + pd.Timedelta(days=1))])
        bars = minute.loc[minute.timestamp.dt.strftime("%H%M").isin(
            ("1452", "1453", "1454", "1455"))].sort_values("timestamp")
        if anchor_lookup:
            earlier = minute.loc[minute.timestamp.dt.strftime("%H%M").isin(
                ("1446", "1447", "1448", "1449"))].sort_values("timestamp")
            old_quote = (earlier.turnover.sum() / earlier.volume.sum()
                         if earlier.timestamp.dt.strftime("%H%M").tolist()
                         == ["1446", "1447", "1448", "1449"]
                         and earlier.volume.sum() > 0 else None)
            expected_quote = anchor_lookup[(row.date, row.code)]
            if (old_quote is None or not np.isfinite(expected_quote)
                    or abs(old_quote - expected_quote) > 1e-8):
                anchor_mismatches.append({"date": row.date, "code": row.code})
        labels = bars.timestamp.dt.strftime("%H%M").tolist()
        quote = (pd.Series({"vwap": bars.turnover.sum() / bars.volume.sum(),
                            "volume": bars.volume.sum()})
                 if labels == ["1452", "1453", "1454", "1455"]
                 and bars.volume.sum() > 0 else None)
        daily = pd.Series({"date": row.date, "tradestatus": row.tradestatus,
                           "preclose": row.preclose, "isST": row.isST})
        def price_and_status(notional: float) -> tuple[float | None, str]:
            shares = _order_shares(row.code, float(quote["vwap"]) if quote is not None
                                   else 0.0, notional)
            if shares == 0:
                return None, "below_minimum_lot"
            return _fill(quote, daily, row.code, "buy", shares,
                         Assumptions(target_notional=notional))
        price, status = price_and_status(100_000)
        low_price, low_status = price_and_status(20_000)
        low_size_fills += low_status == "filled" and low_price is not None
        match = (status == row.entry_status
                 and ((price is None and pd.isna(row.entry_price))
                      or (price is not None and pd.notna(row.entry_price)
                          and abs(price - row.entry_price) < 1e-8)))
        if not match:
            mismatches.append({"date": row.date, "code": row.code,
                               "archived_status": row.entry_status,
                               "raw_status": status,
                               "archived_price": row.entry_price,
                               "raw_price": price})
    result = {"sample_rows": len(sample), "raw_minute_mismatches": len(mismatches),
              "raw_anchor_mismatches": len(anchor_mismatches),
              "low_notional_fill_rate": low_size_fills / len(sample),
              "examples": mismatches[:5], "anchor_examples": anchor_mismatches[:5]}
    (input_dir / "raw_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate", "verify-raw",
                                          "anchor-sensitivity"))
    args = parser.parse_args()
    action = {"freeze": freeze, "evaluate": evaluate,
              "verify-raw": verify_raw_sample,
              "anchor-sensitivity": anchor_sensitivity}[args.stage]
    print(json.dumps(action(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
