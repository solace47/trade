"""Cache point-in-time BaoStock CSRC industry snapshots for recent research.

One query on the first trading day of each quarter supplies that quarter.
Using an earlier classification may be stale, but cannot introduce a future
industry reassignment into a 14:50 signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import baostock as bs
import pandas as pd


def quarter_starts(calendar_file: Path,
                   start: str = "2024-01-01") -> list[str]:
    calendar = pd.read_parquet(calendar_file)
    trading = calendar.loc[
        calendar.is_trading_day.astype(str).eq("1")
        & calendar.calendar_date.between(start, "2025-12-31"),
        "calendar_date",
    ].copy()
    if trading.empty:
        raise ValueError("No recent trading dates in the calendar")
    dates = pd.to_datetime(trading)
    starts = trading.groupby([dates.dt.year, dates.dt.quarter]).min().tolist()
    first = pd.Timestamp(start)
    expected = (2025 - first.year) * 4 + 4 - first.quarter + 1
    if len(starts) != expected:
        raise ValueError("Incomplete quarterly industry as-of dates")
    return starts


def validate_industry(frame: pd.DataFrame, asof: str) -> pd.DataFrame:
    required = {"updateDate", "code", "industry", "industryClassification"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Incomplete BaoStock industry response")
    if frame.code.duplicated().any() or frame.updateDate.gt(asof).any():
        raise ValueError("Duplicate code or future industry update")
    if not frame.industryClassification.eq("证监会行业分类").all():
        raise ValueError("Unexpected industry classification standard")
    result = frame.copy()
    result["asof"] = asof
    return result


def build_intervals(files: list[Path], output: Path) -> pd.DataFrame:
    """Use the last nonempty known class, delaying same-day updates one day."""
    if not files:
        raise ValueError("No point-in-time industry snapshots")
    frames = [validate_industry(pd.read_parquet(path), path.stem)
              for path in sorted(files)]
    history = pd.concat(frames, ignore_index=True)
    history = history.loc[history.industry.ne("")].copy()
    updated = pd.to_datetime(history.updateDate) + pd.Timedelta(days=1)
    queried = pd.to_datetime(history["asof"])
    history["effective_date"] = pd.concat(
        [updated, queried], axis=1
    ).max(axis=1).dt.strftime("%Y-%m-%d")
    history = history.sort_values(["code", "effective_date", "asof"])
    if history.duplicated(["code", "effective_date"]).any():
        raise ValueError("Conflicting industry snapshots on one effective date")
    history["next_effective_date"] = history.groupby("code")[
        "effective_date"
    ].shift(-1)
    result = history[["code", "industry", "asof", "updateDate",
                      "effective_date", "next_effective_date"]].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    return result


def download(calendar_file: Path, output_dir: Path,
             start: str = "2024-01-01") -> list[Path]:
    dates = quarter_starts(calendar_file, start)
    output_dir.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    written = []
    try:
        for asof in dates:
            output = output_dir / f"{asof}.parquet"
            if output.exists():
                validate_industry(pd.read_parquet(output), asof)
                written.append(output)
                continue
            response = bs.query_stock_industry(date=asof)
            if response.error_code != "0":
                raise RuntimeError(f"Industry query failed for {asof}: "
                                   f"{response.error_msg}")
            rows = []
            while response.next():
                rows.append(response.get_row_data())
            frame = validate_industry(
                pd.DataFrame(rows, columns=response.fields), asof
            )
            frame.to_parquet(output, index=False, compression="zstd")
            written.append(output)
            print(f"{asof}: {len(frame)} classifications, "
                  f"{frame.industry.ne('').sum()} nonempty", flush=True)
    finally:
        bs.logout()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"
    ))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/industry_history"))
    parser.add_argument("--intervals", type=Path,
                        default=Path("data/research/industry_intervals.parquet"))
    parser.add_argument("--start", default="2024-01-01",
                        help="First quarter to query; 2023-07-01 adds indicator warmup")
    args = parser.parse_args()
    result = download(args.calendar, args.output_dir, args.start)
    intervals = build_intervals(result, args.intervals)
    print({"snapshots": len(result), "intervals": len(intervals),
           "output": str(args.intervals)})


if __name__ == "__main__":
    main()
