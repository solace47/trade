"""Check how a one-minute earlier cutoff changes a frozen input list.

This reads only the 14:20, 14:49 and 14:50 source bars. It does not
rerank stocks or access any execution or outcome data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(selections_path: Path, minute_root: Path) -> dict:
    selected = pd.read_parquet(selections_path)
    required = {"date", "code", "candidate", "pair_id", "price_1450", "return_last30"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Selection is missing: {sorted(required - set(selected.columns))}")
    if not selected.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Only 2024–2025 frozen selections are permitted")
    if selected.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate selection keys")

    source_rows = []
    for code, group in selected.groupby("code", sort=True):
        exchange, symbol = code.split(".")
        path = minute_root / exchange.upper() / f"{symbol}.parquet"
        if not path.exists():
            continue
        stamps = [pd.Timestamp(f"{date} {label}")
                  for date in group.date.unique()
                  for label in ("14:20:00", "14:49:00", "14:50:00")]
        bars = pd.read_parquet(
            path, columns=["timestamp", "close", "volume"],
            filters=[("timestamp", "in", stamps)],
        )
        if bars.empty:
            continue
        bars["date"] = bars.timestamp.dt.strftime("%Y-%m-%d")
        bars["label"] = bars.timestamp.dt.strftime("%H%M")
        bars["code"] = code
        source_rows.append(bars[["date", "code", "label", "close", "volume"]])
    if not source_rows:
        raise ValueError("No source bars found")
    source = pd.concat(source_rows, ignore_index=True)
    if source.duplicated(["date", "code", "label"]).any():
        raise ValueError("Duplicate source minute label")
    prices = source.pivot(index=["date", "code"], columns="label", values="close")
    prices = prices.rename(columns={label: f"source_{label}" for label in prices})
    volume = source.loc[source.label.eq("1450"), ["date", "code", "volume"]]
    volume = volume.set_index(["date", "code"]).rename(columns={"volume": "volume_1450"})
    checked = selected.join(prices.join(volume), on=["date", "code"])
    columns = ["source_1420", "source_1449", "source_1450"]
    complete = checked[columns].notna().all(axis=1) & checked[columns].gt(0).all(axis=1)
    if (complete & (checked.price_1450 - checked.source_1450).abs().gt(.005)).any():
        raise ValueError("Frozen 14:50 price disagrees with the source minute")
    original_return = checked.source_1450 / checked.source_1420 - 1
    if (complete & (checked.return_last30 - original_return).abs().gt(1e-6)).any():
        raise ValueError("Frozen last-30-minute return disagrees with source bars")
    return _summarize(checked, complete)


def _summarize(checked: pd.DataFrame, complete: pd.Series) -> dict:
    checked = checked.copy()
    checked["complete"] = complete
    checked["earlier_return"] = checked.source_1449 / checked.source_1420 - 1
    checked["retains_role"] = False
    down = checked.candidate.eq("late_decline")
    up = checked.candidate.eq("late_rally_control")
    if not (down | up).all():
        raise ValueError("Unexpected candidate role")
    checked.loc[down, "retains_role"] = (
        complete[down] & checked.loc[down, "earlier_return"].between(-.01, -.003)
    )
    checked.loc[up, "retains_role"] = (
        complete[up] & checked.loc[up, "earlier_return"].between(.003, .01)
    )
    checked["year"] = checked.date.str[:4]
    checked["move_bps"] = (checked.source_1450 / checked.source_1449 - 1).abs() * 10_000
    report = {}
    for year, group in checked.groupby("year", sort=True):
        valid = group.loc[group.complete]
        pairs = group.groupby(["date", "pair_id"]).agg(
            rows=("retains_role", "size"), roles=("candidate", "nunique"),
            both_retain=("retains_role", "all"),
        )
        if pairs.rows.ne(2).any() or pairs.roles.ne(2).any():
            raise ValueError("Frozen list has an incomplete pair")
        report[year] = {
            "selected_rows": len(group),
            "complete_source_rows": len(valid),
            "pairs": len(pairs),
            "pairs_both_roles_at_1449": int(pairs.both_retain.sum()),
            "rows_role_at_1449": int(group.retains_role.sum()),
            "rows_price_changed_in_last_minute": int(valid.move_bps.gt(0).sum()),
            "rows_positive_1450_volume": int(valid.volume_1450.gt(0).sum()),
            "absolute_last_minute_move_bps_median": float(valid.move_bps.median()),
            "absolute_last_minute_move_bps_p95": float(valid.move_bps.quantile(.95)),
        }
    return {
        "scope": "Frozen late-to-open pairs; input sensitivity only, no reranking or outcomes",
        "source_labels": ["14:20", "14:49", "14:50"],
        "by_year": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path,
                        default=Path("data/research/late_to_open_reversal/selections.parquet"))
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/minute_cutoff_audit.json"))
    args = parser.parse_args()
    report = audit(args.selections, args.minute_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
