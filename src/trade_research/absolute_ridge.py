"""Freeze a cash-aware 14:50 ridge ranking before reading test outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .late_to_open_reversal import _board
from .market_study import _quality_keys


ROOT = Path("data/research")
OUTPUT = ROOT / "absolute_ridge"
MAIN_OUTPUT = ROOT / "absolute_ridge_main"
FEATURES = (
    "return_1450", "return_1450_sq", "position_1450", "log_amount",
    "log_volume_ratio", "return5_prior_adjusted",
    "return20_prior_adjusted", "distance_ma20_adjusted",
    "intraday_range", "overnight_gap", "abs_overnight_gap",
    "return_last30", "abs_return_last30", "volume_share_last30",
    "premium_to_last30_vwap", "log_price", "star", "chinext",
)
PERIODS = (
    ("2024-06-14", "2024-07-01", "2024-12-17", "2024H2"),
    ("2024-12-17", "2025-01-01", "2025-12-17", "2025"),
)
CAPACITY = 5
COOLDOWN = 5


def feature_frame(connection: duckdb.DuckDBPyConnection,
                  main_only: bool = False) -> pd.DataFrame:
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    frame = connection.execute("""
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.amount_1450, s.volume_ratio_est, s.return5_prior_adjusted,
               s.return20_prior_adjusted, s.distance_ma20_adjusted,
               s.high_1450, s.low_1450, s.preclose, s.open_1450,
               s.price_1450, i.return_last30, i.volume_share_last30,
               i.premium_to_last30_vwap
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND ABS(s.price_1450 - i.price_1450) <= .005
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
    frame["board"] = frame.code.map(_board)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    frame = frame.loc[frame.board.notna()].copy()
    if main_only:
        frame = frame.loc[frame.board.eq("main")].copy()
    if frame.empty or frame.duplicated(["date", "code"]).any():
        raise ValueError("Invalid 14:50 feature universe")
    return frame


def training_labels(connection: duckdb.DuckDBPyConnection,
                    train: pd.DataFrame, test_first: str) -> pd.DataFrame:
    connection.register("train_keys", train[["date", "code"]])
    labelled = connection.execute("""
        SELECT t.date, t.code, o.exit_date, o.net_return,
               o.exit_status = 'filled' AND o.exit_delay_sessions = 0
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = t.code AND q.date >= t.date
                     AND q.date <= o.exit_date
               ) AS clean_ontime_exit
        FROM train_keys t JOIN outcomes o USING (date, code)
        WHERE o.horizon = 5
    """).df()
    if (len(labelled) != len(train)
            or labelled.duplicated(["date", "code"]).any()
            or labelled.exit_date.ge(test_first).any()):
        raise ValueError("Training labels are incomplete or overlap test time")
    return labelled


def fit(train: pd.DataFrame, labelled: pd.DataFrame) -> tuple[dict, dict]:
    rows = train.merge(labelled, on=["date", "code"], validate="one_to_one")
    if len(rows) != len(train):
        raise ValueError("A training feature lacks an outcome")
    y = rows.net_return.where(rows.clean_ontime_exit, 0.0).fillna(0.0)
    y = y.clip(-.15, .15).to_numpy(dtype=float)
    median = rows[list(FEATURES)].median()
    scale = (rows[list(FEATURES)].quantile(.75)
             - rows[list(FEATURES)].quantile(.25)).clip(lower=.01)
    x = ((rows[list(FEATURES)] - median) / scale).clip(-5, 5).to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered = x - x_mean
    coefficients = np.linalg.solve(
        centered.T @ centered + len(rows) * .05 * np.eye(len(FEATURES)),
        centered.T @ (y - y_mean),
    )
    intercept = y_mean - x_mean @ coefficients
    model = {"median": median, "scale": scale, "coefficients": coefficients,
             "intercept": float(intercept)}
    audit = {"train_rows": len(rows), "train_signal_last": rows.date.max(),
             "train_exit_last": labelled.exit_date.max(),
             "mean_training_cash_clipped": y_mean,
             "train_clean_exit_fraction": float(rows.clean_ontime_exit.mean()),
             "intercept": float(intercept),
             "coefficients": dict(zip(FEATURES, coefficients.tolist(), strict=True))}
    return model, audit


def score(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    result = frame.copy()
    x = ((result[list(FEATURES)] - model["median"])
         / model["scale"]).clip(-5, 5).to_numpy(dtype=float)
    result["score"] = x @ model["coefficients"] + model["intercept"]
    return result


def choose(scored: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    date_index = {date: index for index, date in enumerate(calendar)}
    last_selected: dict[str, int] = {}
    selected = []
    ordered = scored.loc[scored.score.gt(0)].sort_values(
        ["date", "score", "code"], ascending=[True, False, True])
    for date, group in ordered.groupby("date", sort=True):
        index = date_index[date]
        rank = 0
        for row in group.itertuples(index=False):
            if rank >= CAPACITY:
                break
            if index - last_selected.get(row.code, -1000) <= COOLDOWN:
                continue
            rank += 1
            selected.append({"date": date, "code": row.code,
                             "daily_rank": rank, "score": row.score})
            last_selected[row.code] = index
    return pd.DataFrame(selected)


def match_controls(chosen: pd.DataFrame,
                   features: pd.DataFrame) -> pd.DataFrame:
    selected_keys = set(zip(chosen.date, chosen.code))
    controls = []
    for date, group in chosen.groupby("date", sort=True):
        available = features.loc[features.date.eq(date)].copy()
        available = available.loc[~available.code.map(
            lambda code: (date, code) in selected_keys
        )].set_index("code", drop=False)
        chosen_features = group.merge(features, on=["date", "code"],
                                      validate="one_to_one")
        for row in chosen_features.sort_values("daily_rank").itertuples(index=False):
            fresh = available.loc[available.board.eq(row.board)]
            amount_ratio = fresh.amount_1450 / row.amount_1450
            price_ratio = fresh.price_1450 / row.price_1450
            prior_gap = (fresh.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            day_gap = (fresh.return_1450 - row.return_1450).abs()
            matched = fresh.loc[
                amount_ratio.between(.5, 2) & price_ratio.between(.5, 2)
                & prior_gap.le(.05) & day_gap.le(.02)
            ].copy()
            if matched.empty:
                continue
            matched["distance"] = (
                prior_gap.loc[matched.index] / .05
                + day_gap.loc[matched.index] / .02
                + np.abs(np.log(amount_ratio.loc[matched.index])) / np.log(2)
                + np.abs(np.log(price_ratio.loc[matched.index])) / np.log(2)
            )
            control = matched.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            controls.append({"date": date, "code": control.code,
                             "pair_id": row.code, "daily_rank": row.daily_rank,
                             "distance": control.distance})
            available = available.drop(index=control.code)
    return pd.DataFrame(controls, columns=[
        "date", "code", "pair_id", "daily_rank", "distance",
    ])


def freeze(output_dir: Path = OUTPUT, main_only: bool = False) -> dict:
    if main_only and output_dir == OUTPUT:
        raise ValueError("Main-board retraining needs a separate output directory")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    features = feature_frame(connection, main_only=main_only)
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    chosen_frames, training_audits = [], []
    for train_end, test_first, test_last, name in PERIODS:
        train = features.loc[features.date.le(train_end)].copy()
        labels = training_labels(connection, train, test_first)
        model, model_audit = fit(train, labels)
        test = features.loc[features.date.between(test_first, test_last)]
        scored = score(test, model)
        picks = choose(scored, calendar)
        if not picks.empty:
            picks["test_period"] = name
            chosen_frames.append(picks)
        model_audit.update({"test_period": name, "test_rows": len(test),
                            "positive_predictions": int(scored.score.gt(0).sum()),
                            "selected": len(picks)})
        training_audits.append(model_audit)
    if not chosen_frames:
        raise ValueError("Absolute model produced no positive-score candidates")
    chosen = pd.concat(chosen_frames, ignore_index=True)
    controls = match_controls(chosen, features)
    chosen["candidate"] = "absolute_model"
    chosen["pair_id"] = chosen.code
    controls["candidate"] = "same_day_control"
    membership = pd.concat([chosen, controls], ignore_index=True)
    if (membership.duplicated(["date", "code"]).any()
            or not membership.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid frozen absolute-model memberships")
    control_keys = set(zip(controls.date, controls.pair_id))
    chosen["matched_control"] = [
        (date, code) in control_keys for date, code in zip(chosen.date, chosen.code)
    ]
    period = chosen.date.str[:4] + "H" + np.where(
        chosen.date.str[5:7].astype(int) <= 6, "1", "2")
    by_half = chosen.assign(half=period).groupby("half").agg(
        signals=("code", "size"), signal_days=("date", "nunique"),
        matched_controls=("matched_control", "sum"),
    ).reset_index().to_dict("records")
    by_board = chosen.assign(board=chosen.code.map(_board)).groupby("board").agg(
        signals=("code", "size"), matched_controls=("matched_control", "sum"),
    ).reset_index().to_dict("records")
    audit = {
        "feature_rows": len(features), "training": training_audits,
        "universe": "main" if main_only else "all_boards",
        "signals": len(chosen), "controls": len(controls),
        "control_fraction": len(controls) / len(chosen),
        "by_half": by_half, "by_board": by_board,
        "capacity": CAPACITY, "cooldown": COOLDOWN,
        "score_threshold": 0.0, "treatment_label": "absolute_model",
        "control_label": "same_day_control",
    }
    audit["outcome_gate_passed"] = (
        len(by_half) == 3
        and all(item["signal_days"] >= 40 and item["signals"] >= 100
                for item in by_half)
        and audit["control_fraction"] >= .70
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    membership.to_parquet(output_dir / "selections.parquet", index=False,
                          compression="zstd")
    if audit["outcome_gate_passed"]:
        connection.register("members", membership[["date", "code"]])
        snapshots = connection.execute("""
            SELECT s.* FROM members m JOIN snapshots s USING (date, code)
        """).df()
        if len(snapshots) != len(membership):
            raise ValueError("A frozen selection lacks an original snapshot")
        snapshots.to_parquet(output_dir / "repricing_signals.parquet",
                             index=False, compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    output = args.output or (MAIN_OUTPUT if args.main_only else OUTPUT)
    report = freeze(output, main_only=args.main_only)
    print({key: report[key] for key in (
        "signals", "controls", "control_fraction", "by_half",
        "outcome_gate_passed")})


if __name__ == "__main__":
    main()
