"""Freeze complete risk-warning removals and their first feasible tail entry."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import match_controls
from .corporate_cash import save_json, sha
from .hf_outcomes import _limit_price
from .quote_precision import fixed_quote_shares, quote_cents
from .turnover_reference import CALENDAR

ROOT = Path("data/research/risk_removal_1449")
RULE_COMMIT = "8484ec0"
REVIEWS = Path("config/risk_removal_source_reviews.json")
DATE = r"(202\d年\d{1,2}月\d{1,2}日)"
EFFECTIVE = re.compile(
    r"撤销[^。；;]{0,22}起始日[：:为]*(?:自)?" + DATE
    + r"|(?:公司股票(?:交易)?|公司)自" + DATE
    + r"(?:[（(][^）)]{0,6}[）)])?(?:开市)?(?:起|日起)(?:被)?撤销")


def document_fields(text: str) -> dict:
    text = re.sub(r"\s+", "", text)
    dates = sorted({pd.Timestamp(re.sub(r"[年月]", "-", (a or b)).replace("日", "")).date().isoformat()
                    for a, b in EFFECTIVE.findall(text)})
    names = re.findall(r"(?:简称由|简称：由|简称将由)[“「]?[*＊]?ST[^”」]{1,10}[”」]"
                       r"(?:变更为|变更为：|变更为:|改为)[“「]([^”」]+)", text)
    if not names:
        names = re.findall(r"撤销后A股简称为([^。；，\uf06c]+)", text)
    normal = [name for name in names if not re.search(r"ST|退|[*＊]", name)]
    return {"effective_dates": dates, "normal_names": sorted(set(normal)),
            "normal_limit": "10%" in text or "10％" in text}


def verify_sources(output: Path = ROOT) -> pd.DataFrame:
    """Require matching original hashes and two independent text extractions."""
    manifest = json.loads((output / "formal_source_manifest.json").read_text())
    reviews = json.loads(REVIEWS.read_text())
    rows, checks = [], []
    for row in manifest:
        pdf, extracted = Path(row["pdf_path"]), Path(row["text_path"])
        if sha(pdf) != row["pdf_sha256"] or sha(extracted) != row["text_sha256"]:
            raise ValueError("Risk-removal original or extraction changed")
        independent = pdf.with_suffix(".pypdf.json")
        texts = ["".join(page["text"] for page in json.loads(extracted.read_text())),
                 "".join(json.loads(independent.read_text()))]
        fields = [document_fields(text) for text in texts]
        if fields[0] != fields[1]:
            raise ValueError(f"Independent document fields disagree: {row['code']}")
        field = fields[0]
        if (not field["normal_limit"] or len(field["normal_names"]) != 1
                or any(row["code"][3:] not in re.sub(r"\s+", "", text) for text in texts)):
            raise ValueError("Missing complete-removal identity, name or normal limit")
        exception = reviews.get(row["code"] + ":" + row["event_date"])
        if field["effective_dates"] != [row["event_date"]]:
            if (not exception or exception["pdf_sha256"] != row["pdf_sha256"]
                    or exception["effective_date"] != row["event_date"]
                    or exception["observed_dates"] != field["effective_dates"]):
                raise ValueError("Conflicting operative dates need an exact-source review")
        if not "2024-01-01" <= row["notice_date"] < row["event_date"] <= "2025-12-31":
            raise ValueError("Notice must precede the historical effective date")
        rows.append({**row, "normal_name": field["normal_names"][0], "source_exception": bool(exception)})
        checks.append({"code": row["code"], "event_date": row["event_date"], **field,
            "pypdf_sha256": sha(independent), "source_exception": exception})
    result = pd.DataFrame(rows)
    if result.duplicated(["event_date", "code"]).any():
        raise ValueError("Duplicate risk-removal source identity")
    cached = pd.read_parquet(output / "formal_notice_candidates.parquet")
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), result[cached.columns].reset_index(drop=True))
    save_json(output / "source_verification.json", {"manifest_sha256": sha(output / "formal_source_manifest.json"),
        "reviews_sha256": sha(REVIEWS), "checks": checks})
    result.to_parquet(output / "verified_events.parquet", index=False, compression="zstd")
    return result


def decision_buyable(code: str, price: float, preclose: float) -> bool:
    if not code.startswith(("sh.60", "sz.00")) or not np.isfinite(preclose) or preclose <= 0:
        return False
    try:
        quote = quote_cents(price) / 100
        shares = fixed_quote_shares(code, quote, 20000)
    except ValueError:
        return False
    # Decimal avoids admitting an exact half-cent boundary through float noise.
    exact = Decimal(quote_cents(price)) / 100
    pressure = max(exact * Decimal(".0015"), Decimal(".005"))
    upper = Decimal(str(_limit_price(preclose, .1, True)))
    return shares >= 100 and exact + pressure < upper - Decimal(".005")


def causal_exclusions(pool: pd.DataFrame, events: pd.DataFrame,
                      notices: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions = {date: i for i, date in enumerate(calendar)}
    event_positions = events.assign(position=events.event_date.map(positions)).groupby("code").position.agg(list).to_dict()
    delist = notices.loc[notices.title.str.contains(r"进入退市整理|退市整理期交易", regex=True)]
    delist_dates = delist.groupby("code").notice_date.min().to_dict()
    result = pool.copy()
    result["recent_removal"] = [any(0 <= positions[d] - p <= 20 for p in event_positions.get(code, []))
                                for d, code in zip(result.date, result.code)]
    # Titles establish known delisting risk, not an exact start date or future exclusion.
    result["known_delisting"] = [code in delist_dates and delist_dates[code] < d
                                 for d, code in zip(result.date, result.code)]
    return result


def watch_schedule(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    if calendar != sorted(set(calendar)):
        raise ValueError("Calendar must be unique and ordered")
    positions = {date: i for i, date in enumerate(calendar)}
    rows = []
    for row in events.itertuples():
        start = positions[row.event_date]
        for delay in range(5):
            i = start + delay
            if i >= len(calendar):
                continue
            complete = i + 10 < len(calendar) and calendar[i + 10] <= "2025-12-31"
            rows.append({"date": calendar[i], "code": row.code, "event_date": row.event_date,
                         "entry_delay": delay, "complete_observation": complete})
    return pd.DataFrame(rows)


def visible_pool(dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    if not dates or any(not "2024-01-01" <= d <= "2025-12-31" for d in dates):
        raise ValueError("Only historical exploration inputs are allowed")
    prefixes = sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet"))
    snapshots = sorted(Path("data/research/market_snapshots_ci").glob("*.parquet"))
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.read_parquet([str(p) for p in prefixes]).create_view("prefix")
    c.read_parquet([str(p) for p in snapshots]).create_view("snapshots")
    c.register("decision_dates", pd.DataFrame({"date": sorted(set(dates))}))
    joined = c.execute("""SELECT p.date,p.code,p.price_1449,p.amount_1449,p.volume_1449,
            p.high_1449,p.low_1449,p.quote_outside_traded_range,
            s.preclose,s.isST,s.listing_age_sessions,s.reference_gap,s.return20_prior_adjusted
        FROM prefix p JOIN decision_dates d USING(date)
        LEFT JOIN snapshots s USING(date,code) ORDER BY p.date,p.code""").df()
    c.close()
    if joined.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate visible stock-day")
    if joined.loc[joined.amount_1449.ge(1e8), "isST"].isna().any():
        raise ValueError("An amount-eligible stock-day lacks historical state")
    joined["necessary_buyable"] = [decision_buyable(r.code, r.price_1449, r.preclose)
                                   for r in joined.itertuples()]
    joined["eligible"] = (joined.code.str.startswith(("sh.60", "sz.00")) & joined.isST.eq(0)
        & joined.listing_age_sessions.ge(20) & joined.reference_gap.eq(False)
        & joined.quote_outside_traded_range.eq(False) & np.isfinite(joined.return20_prior_adjusted)
        & joined.amount_1449.ge(1e8) & joined.volume_1449.gt(0) & joined.necessary_buyable)
    frame = joined.loc[joined.eligible].copy()
    frame["raw_price_1449"] = frame.price_1449
    frame["price_1449"] = frame.price_1449.map(lambda p: quote_cents(p) / 100)
    frame["return_1450"] = frame.price_1449 / frame.preclose - 1  # Legacy matching name only.
    frame["board"], frame["price_signal"], frame["amount_signal"] = "main", frame.price_1449, frame.amount_1449
    return frame, joined, prefixes + snapshots


def first_entries(watch: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    eligible = watch.loc[watch.complete_observation].merge(pool, on=["date", "code"], validate="many_to_one")
    first = eligible.sort_values(["event_date", "code", "date"]).drop_duplicates(["event_date", "code"])
    first = first.sort_values(["date", "amount_1449", "code"], ascending=[True, False, True]).copy()
    first["daily_rank"] = first.groupby("date").cumcount() + 1
    return first


def freeze(output: Path = ROOT) -> dict:
    if (output / "repriced.parquet").exists() or (output / "continued/repriced.parquet").exists():
        raise ValueError("Cannot replace event inputs after reading their outcomes")
    events = verify_sources(output)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    watch = watch_schedule(events, calendar)
    pool, inputs, sources = visible_pool(watch.loc[watch.complete_observation, "date"].unique().tolist())
    pool = causal_exclusions(pool, events, pd.read_parquet(output / "notice_index.parquet"), calendar)
    first = first_entries(watch, pool.loc[~pool.known_delisting])
    chosen = first.loc[first.daily_rank.le(5)].copy()
    chosen_keys = set(zip(chosen.date, chosen.code))
    control_pool = pool.loc[(~pool.known_delisting & ~pool.recent_removal)
                           | pd.Series([(d, c) in chosen_keys for d, c in zip(pool.date, pool.code)], index=pool.index)]
    controls = match_controls(chosen[["date", "code", "daily_rank"]], control_pool)
    controls = controls.merge(pool, on=["date", "code"], validate="one_to_one")
    chosen["arm"], chosen["pair_id"] = "high", chosen.code
    controls["arm"] = "low"
    signals = pd.concat([chosen, controls], ignore_index=True).sort_values(["date", "arm", "daily_rank", "code"])
    signals["pair_id"] = signals.date + ":" + signals.pair_id
    signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    if signals.duplicated(["date", "code"]).any() or controls[["recent_removal", "known_delisting"]].any().any():
        raise ValueError("Invalid event or control identities")
    audit = watch.merge(inputs, on=["date", "code"], how="left", validate="many_to_one")
    for name, frame in (("watch_schedule", watch), ("event_input_audit", audit), ("eligible_pool", pool),
                        ("first_entries", first), ("signals", signals)):
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    save_json(output / "feature_sources.json", {str(path): sha(path) for path in sources})
    report = {"rule_commit": RULE_COMMIT, "events": len(events), "candidates": len(chosen), "controls": len(controls),
        "over_daily_cap": int(first.daily_rank.gt(5).sum()), "no_feasible_entry": len(events) - len(first),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), dates=("date", "nunique"))
            .reset_index().to_dict("records"),
        "entry_delays": chosen.entry_delay.value_counts().sort_index().to_dict(),
        "below_five_yuan": int(chosen.price_1449.lt(5).sum()),
        "above_old_amount_ceiling": int(chosen.amount_1449.gt(1e9).sum()),
        "signals_sha256": sha(output / "signals.parquet"),
        "inputs_sha256": {name: sha(output / name) for name in ("eligible_pool.parquet", "verified_events.parquet",
            "formal_source_manifest.json", "source_verification.json", "notice_index.parquet", "feature_sources.json")},
        "calendar_sha256": sha(CALENDAR), "notional": 20000, "new_execution_outcomes_read": False,
        "holdout_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
