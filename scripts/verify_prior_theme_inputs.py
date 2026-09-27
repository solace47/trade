"""Independently reconstruct date proof, prior states, theme choice and context."""
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

ROOT = Path("data/research/prior_theme_broken")


def check(ROOT=ROOT):
    report = json.loads((ROOT / "input_report.json").read_text())
    hist = json.loads((ROOT / "history_report.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert report["history_report_sha256"] == sha(ROOT / "history_report.json")
    for name, digest in report["output_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, digest in {**hist["source_sha256"], **report["source_sha256"]}.items():
        assert sha(Path(name)) == digest, name
    assert report["date_proof_sha256"] == sha(Path("config/prior_theme_date_proof.json"))
    assert sha(Path("config/prior_theme_broken_protocol.json")) == manifest["protocol_sha256"]
    assert sha(ROOT / "source_jobs.parquet") == manifest["jobs_sha256"]
    assert sha(ROOT / "visible_main.parquet") == manifest["main_sha256"]
    assert hist["prior_state_sha256"] == sha(ROOT / "prior_state.parquet")
    jobs = pd.read_parquet(ROOT / "source_jobs.parquet")
    calendar = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
    dates = sorted(calendar.loc[calendar.is_trading_day.eq("1"), "calendar_date"])
    positions = {d: i for i, d in enumerate(dates)}
    assert len(jobs) == manifest["source_dates"] and not jobs.date.duplicated().any()
    assert all(positions[d] - positions[p] == 1 for d, p in jobs.itertuples(index=False, name=None))
    assert jobs.source_date.between("2024-01-02", "2025-12-29").all()
    assert jobs.date.between("2024-01-03", "2025-12-30").all()
    paths = [p for p in hist["source_sha256"] if "/daily/" in p]
    c = duckdb.connect()
    c.register("wanted", jobs)
    c.read_parquet(paths).create_view("raw")
    raw = c.sql("""SELECT w.date,w.source_date,r.code,r.open,r.high,r.low,r.close,r.preclose,
        r.volume,r.amount,r.tradestatus,r.isST,r.adjustflag
        FROM raw r JOIN wanted w ON r.date=w.source_date ORDER BY w.date,r.code""").df()
    actual = pd.read_parquet(ROOT / "prior_state.parquet").sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(raw, actual[raw.columns], check_dtype=False, check_exact=True)
    basic = pd.read_parquet("data/baostock/market_2020_2026/metadata/stock_basic.parquet", columns=["code", "ipoDate"]).set_index("code").ipoDate
    ipo = raw.code.map(basic)
    ages = raw.source_date.map(positions).to_numpy() - np.searchsorted(np.array(dates), ipo.fillna("9999-12-31")) + 1
    np.testing.assert_array_equal(actual.age, ages)
    prices = raw[["open", "high", "low", "close", "preclose"]].to_numpy()
    cents = np.rint(prices * 100).astype("int64")
    upper = (cents[:, 4] * 11 + 5) // 10
    lower = (cents[:, 4] * 9 + 5) // 10
    good = (raw.tradestatus.eq(1) & raw.isST.eq(0) & raw.adjustflag.eq(3) & (ages >= 20)
            & ipo.notna() & ipo.le(raw.source_date) & raw.volume.gt(0) & raw.amount.gt(0))
    good &= (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
             & (np.abs(prices - cents / 100) <= .0001).all(axis=1)
             & (raw.high + .0001 >= prices[:, :4].max(axis=1))
             & (raw.low - .0001 <= np.minimum(raw.open, raw.close))
             & (raw.high * 100 <= upper + .0001) & (raw.low * 100 >= lower - .0001))
    status = np.select([~good, cents[:, 3] == upper, cents[:, 1] == upper],
                       ["unknown", "closed_limit", "broken_limit"], default="other")
    np.testing.assert_array_equal(actual.prior_eligible, good)
    np.testing.assert_array_equal(actual.prior_status, status)
    np.testing.assert_array_equal(actual.upper_cents, upper)
    np.testing.assert_array_equal(actual.lower_cents, lower)
    # Scalar decimal anchors check the actual exchange-cent rounding separately.
    anchors = actual.assign(order=(actual.source_date + actual.code).map(
        lambda s: __import__("hashlib").sha256(s.encode()).hexdigest())).sort_values("order").head(96)
    for a in anchors.itertuples(index=False):
        assert int((Decimal(str(a.preclose)) * Decimal("1.1") * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)) == a.upper_cents
        assert int((Decimal(str(a.preclose)) * Decimal("0.9") * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)) == a.lower_cents
    state = {(d, code): s for d, code, s in zip(raw.date, raw.code, status)}
    cash = {(d, code): Decimal(str(amount)) for d, code, amount in zip(raw.source_date, raw.code, raw.amount)}
    sources, relations, raw_relations, audit_counts = {}, set(), Counter(), []
    for job in jobs.itertuples(index=False):
        meta = json.loads((ROOT / "source" / (job.source_date + ".json")).read_text())
        ok = False
        result_status = meta["status"]
        n_amount = n_conflict = n_mismatch = 0
        if result_status in ["historical_response_received", "date_echo_missing_or_mismatched"]:
            data = json.loads((ROOT / "source" / (job.source_date + ".body")).read_text())["data"]
            assert data["today"] is False
            echo = {str(x["next_day"]) for group in data["plate_stocks"].values() for x in group if x.get("next_day")}
            if echo in [set(), {job.date}] and len(data["plate"]) < 100:
                amt = defaultdict(set)
                for category in ["plate_stocks", "plate_stocks_zb", "plate_stocks_bx"]:
                    for plate, records in data[category].items():
                        assert str(plate).isdigit()
                        for x in records:
                            code = str(x["stock_code"])
                            assert code.isdigit() and len(code) == 6
                            if code[:2] not in ["60", "00"]:
                                continue
                            code = ("sh." if code[:2] == "60" else "sz.") + code
                            relations.add((job.date, job.source_date, code, str(plate)))
                            raw_relations[(job.date, job.source_date, code, str(plate), category)] += 1
                            if x.get("amount") is not None:
                                amt[code].add(Decimal(str(x["amount"])))
                n_amount = len(amt)
                n_conflict = sum(len(v) - 1 for v in amt.values())
                for code, values in amt.items():
                    local = cash.get((job.source_date, code))
                    for value in values:
                        if not value.is_finite() or value <= 0 or local is None or not local.is_finite() or abs(value * 100000000 - local) > Decimal("500000.02"):
                            n_mismatch += 1
                ok = n_amount >= 5 and n_conflict == 0 and n_mismatch == 0
                result_status = ("amount_and_echo_verified" if echo else "amount_verified_echo_absent") if ok else "amount_date_proof_failed"
            elif len(data["plate"]) >= 100 and not echo:
                result_status = "possible_topic_limit"
        sources[job.date] = ok
        audit_counts.append((job.date, ok, result_status, n_amount, n_conflict, n_mismatch))
    audit = pd.read_parquet(ROOT / "source_audit.parquet")
    observed = list(audit[["date", "source_valid", "proof_status", "amount_stocks", "amount_conflicts", "amount_mismatches"]].itertuples(index=False, name=None))
    assert observed == audit_counts
    recorded = pd.read_parquet(ROOT / "raw_relations.parquet")
    assert Counter(recorded[["date", "source_date", "code", "plate_id", "source_category"]].itertuples(index=False, name=None)) == raw_relations
    recorded = pd.read_parquet(ROOT / "relations.parquet")
    assert set(recorded.itertuples(index=False, name=None)) == relations and len(recorded) == len(relations)
    visible = pd.read_parquet(ROOT / "visible_main.parquet")
    actual = pd.read_parquet(ROOT / "features.parquet")
    pd.testing.assert_frame_equal(actual[visible.columns], visible, check_exact=True)
    values = dict(zip(zip(visible.date, visible.code), visible.return_1449))
    return_recomputed = visible.price_1449 / visible.preclose - 1
    np.testing.assert_allclose(visible.return_1449, return_recomputed, atol=1e-15, rtol=0)
    benchmark = visible.groupby("date").return_1449.mean().to_dict()
    market_size = visible.groupby("date").size().to_dict()
    membership, groups = defaultdict(set), defaultdict(set)
    for date, previous, code, theme in relations:
        if sources[date]:
            membership[(date, code)].add(theme)
            if state.get((date, code)) == "closed_limit":
                groups[(date, theme)].add(code)
    # Selection uses ONLY prior membership counts. Current observations are read
    # strictly after the largest prior group (then lexicographic ID) is selected.
    expected = []
    for date, code in zip(visible.date, visible.code):
        prior = state.get((date, code), "unknown")
        themes = membership.get((date, code), set())
        theme = min(themes, key=lambda t: (-len(groups[(date, t)] - {code}), t)) if themes else None
        own = n = k = total = coverage = premium = excess = None
        context = "unknown"
        if theme is not None:
            peers = groups[(date, theme)] - {code}
            own = code in groups[(date, theme)]
            n = len(peers)
            r = [values[(date, p)] for p in sorted(peers) if (date, p) in values]
            k = len(r)
            total = sum(r)
            coverage = k / n if n else None
            if n >= 3 and coverage >= .9:
                premium = total / k
                excess = premium - benchmark[date]
                context = "strong" if excess >= .01 else "weak" if excess <= -.01 else "neutral"
        reason = ("source_unknown" if not sources[date] else "no_reported_theme" if theme is None
                  else "too_few_peers" if n < 3 else "peer_visibility_insufficient" if coverage < .9 else "known")
        expected.append((prior, theme, own, n, k, total, coverage, premium, excess, context, reason))
    columns = ["prior_status", "selected_theme", "self_member", "peer_count", "peer_visible", "peer_return_sum",
               "context_coverage", "peer_return", "excess_premium", "context", "context_reason"]
    expected = pd.DataFrame(expected, columns=columns)
    for name in columns:
        if name in ["peer_count", "peer_visible", "peer_return_sum", "context_coverage", "peer_return", "excess_premium"]:
            np.testing.assert_allclose(actual[name].to_numpy(dtype=float, na_value=np.nan), expected[name].to_numpy(dtype=float, na_value=np.nan),
                                       atol=2e-13, rtol=1e-12, equal_nan=True, err_msg=name)
        else:
            left = actual[name].astype(object).where(actual[name].notna(), "__missing__").astype(str)
            right = expected[name].astype(object).where(expected[name].notna(), "__missing__").astype(str)
            pd.testing.assert_series_equal(left, right, check_names=False)
    np.testing.assert_allclose(actual.market_return, actual.date.map(benchmark), atol=1e-15)
    np.testing.assert_array_equal(actual.market_count, actual.date.map(market_size))
    primary = visible.necessary_tradeable & expected.prior_status.eq("broken_limit") & expected.context.eq("strong")
    np.testing.assert_array_equal(actual.primary, primary)
    # Fields quarantined from prediction may be arbitrarily corrupted without
    # changing the extraction of any identity or contemporaneous amount audit.
    from trade_research.prior_theme_broken_inputs import extract_identities
    mutations = 0
    for job in jobs.iloc[::40].itertuples(index=False):
        payload = json.loads((ROOT / "source" / (job.source_date + ".body")).read_text())
        if not isinstance(payload.get("data"), dict):
            continue
        data = payload["data"]
        before = extract_identities(data, job.date, job.source_date)
        for category in ["plate_stocks", "plate_stocks_zb", "plate_stocks_bx"]:
            for group in data[category].values():
                for row in group:
                    for key in list(row):
                        if key not in ["stock_code", "amount"]:
                            row[key] = "FUTURE_FIELD_CORRUPTED"
        data["stock_info"] = {"future": "invalid"}
        data["plate"] = [["future", "invalid", 1e30]]
        assert extract_identities(data, job.date, job.source_date) == before
        mutations += 1
    result = dict(passed=True, input_report_sha256=sha(ROOT / "input_report.json"), rows=len(actual),
                  raw_daily_rows=len(raw), raw_relations=sum(raw_relations.values()), unique_relations=len(relations),
                  source_dates=len(audit_counts), source_valid_dates=sum(sources.values()), primary=int(primary.sum()),
                  prior_state_fields_checked=int(len(raw) * 4), full_context_fields_checked=int(len(actual) * len(columns)),
                  decimal_rounding_anchors=len(anchors), quarantined_field_mutations=mutations,
                  outputs_not_profit=True, prices_2026_read=False)
    save_json(ROOT / "input_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
