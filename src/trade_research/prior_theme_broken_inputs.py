"""Prior-session identities and present 14:49 theme support, without outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .prior_theme_broken import ROOT, PROTOCOL, CALENDAR, inspect_payload

DATE_PROOF = Path("config/prior_theme_date_proof.json")
DAILY = Path("data/baostock/market_2020_2026/daily")
BASIC = Path("data/baostock/market_2020_2026/metadata/stock_basic.parquet")
MAPS = ("plate_stocks", "plate_stocks_zb", "plate_stocks_bx")


def freeze_check():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for path, key in [(PROTOCOL, "protocol_sha256"), (CALENDAR, "calendar_sha256"),
                      (ROOT / "source_jobs.parquet", "jobs_sha256"),
                      (ROOT / "visible_main.parquet", "main_sha256")]:
        assert sha(path) == manifest[key], (str(path), key)
    return manifest


def history():
    if (ROOT / "history_report.json").exists():
        raise ValueError("Do not replace frozen prior-day state")
    freeze_check()
    paths = sorted(DAILY.glob("sh_60*.parquet")) + sorted(DAILY.glob("sz_00*.parquet"))
    calendar = pd.read_parquet(CALENDAR)
    days = np.array(sorted(calendar.loc[calendar.is_trading_day.eq("1"), "calendar_date"]))
    basic = pd.read_parquet(BASIC, columns=["code", "ipoDate"]).drop_duplicates()
    assert not basic.code.duplicated().any()
    basic["ipo_session_index"] = np.searchsorted(days, basic.ipoDate.fillna("9999-12-31"))
    jobs = pd.read_parquet(ROOT / "source_jobs.parquet")
    jobs["source_session_index"] = np.searchsorted(days, jobs.source_date)
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.register("jobs", jobs)
    c.register("basic", basic)
    c.read_parquet([str(p) for p in paths]).create_view("raw")
    c.execute("""CREATE TABLE history AS SELECT j.date,j.source_date,d.code,
        d.open,d.high,d.low,d.close,d.preclose,d.volume,d.amount,d.tradestatus,d.isST,d.adjustflag,
        b.ipoDate,j.source_session_index-b.ipo_session_index+1 AS age,
        round(d.preclose*100)::BIGINT AS preclose_cents,
        round(d.close*100)::BIGINT AS close_cents,round(d.high*100)::BIGINT AS high_cents,
        (round(d.preclose*100)::BIGINT*110+50)//100 AS upper_cents,
        (round(d.preclose*100)::BIGINT*90+50)//100 AS lower_cents
        FROM raw d JOIN jobs j ON d.date=j.source_date LEFT JOIN basic b USING(code)
        WHERE d.date BETWEEN '2024-01-02' AND '2025-12-29'""")
    c.execute("""CREATE TABLE prior_state AS WITH checked AS (
        SELECT *, (tradestatus=1 AND isST=0 AND adjustflag=3 AND age>=20
          AND ipoDate IS NOT NULL AND ipoDate<=source_date AND volume>0 AND amount>0
          AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(preclose)
          AND least(open,high,low,close,preclose)>0
          AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
          AND greatest(abs(open-round(open,2)),abs(high-round(high,2)),abs(low-round(low,2)),
            abs(close-round(close,2)),abs(preclose-round(preclose,2)))<=.0001
          AND high*100<=upper_cents+.0001 AND low*100>=lower_cents-.0001) IS TRUE AS prior_eligible
        FROM history)
        SELECT *,CASE WHEN NOT prior_eligible THEN 'unknown'
          WHEN close_cents=upper_cents THEN 'closed_limit'
          WHEN high_cents=upper_cents THEN 'broken_limit' ELSE 'other' END AS prior_status
        FROM checked""")
    states = c.sql("SELECT * FROM prior_state ORDER BY date,code").df()
    assert not states.duplicated(["date", "code"]).any()
    assert states.source_date.lt(states.date).all()
    states.to_parquet(ROOT / "prior_state.parquet", index=False, compression="zstd")
    result = dict(manifest_sha256=sha(ROOT / "manifest.json"),
                  date_proof_sha256=sha(DATE_PROOF), prices_2026_read=False, outcomes_read=False,
                  rows=len(states), status_counts=states.prior_status.value_counts().to_dict(),
                  source_sha256={str(p): sha(p) for p in [CALENDAR, BASIC, *paths]},
                  prior_state_sha256=sha(ROOT / "prior_state.parquet"))
    save_json(ROOT / "history_report.json", result)
    return {k: v for k, v in result.items() if k != "source_sha256"}


def extract_identities(data, signal_date, source_date):
    """Explicit field whitelist: future/opaque fields cannot affect identities."""
    relations, amounts = [], []
    for category in MAPS:
        for plate_id, rows in data[category].items():
            if not re.fullmatch(r"[0-9]+", str(plate_id)) or not isinstance(rows, list):
                raise ValueError("Malformed historical membership map")
            for row in rows:
                raw_code = str(row["stock_code"])
                if not re.fullmatch(r"[0-9]{6}", raw_code):
                    raise ValueError("Malformed source security identity")
                # The fixed mechanism concerns ordinary SH/SZ main-board shares.
                if not raw_code.startswith(("60", "00")):
                    continue
                code = ("sh." if raw_code.startswith("60") else "sz.") + raw_code
                relations.append(dict(date=signal_date, source_date=source_date, code=code,
                                      plate_id=str(plate_id), source_category=category))
                value = row.get("amount")
                if value is not None:
                    amounts.append(dict(date=signal_date, source_date=source_date, code=code,
                                        source_amount_yi=float(value)))
    return relations, amounts


def inputs():
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not overwrite frozen theme inputs")
    manifest = freeze_check()
    hr = json.loads((ROOT / "history_report.json").read_text())
    assert hr["prior_state_sha256"] == sha(ROOT / "prior_state.parquet")
    assert hr["date_proof_sha256"] == sha(DATE_PROOF)
    dr = json.loads((ROOT / "download_report.json").read_text())
    assert dr["all_dates_attempted"], "All original dates must be attempted, without resampling"
    jobs = pd.read_parquet(ROOT / "source_jobs.parquet")
    states = pd.read_parquet(ROOT / "prior_state.parquet")
    daily_amounts = states.set_index(["source_date", "code"]).amount
    audit_rows, relations, amounts, provenance = [], [], [], {}
    for job in jobs.itertuples(index=False):
        meta = ROOT / "source" / (job.source_date + ".json")
        body = meta.with_suffix(".body")
        assert sha(meta) == dr["receipts_sha256"][job.source_date]
        receipt = json.loads(meta.read_text())
        assert receipt["source_date"] == job.source_date and receipt["signal_date"] == job.date
        provenance[str(meta)] = sha(meta)
        row = dict(date=job.date, source_date=job.source_date, original_status=receipt["status"],
                   source_valid=False, proof_status=receipt["status"], amount_stocks=0,
                   amount_mismatches=0, amount_conflicts=0, max_amount_difference=None)
        if receipt.get("body_sha256"):
            assert sha(body) == receipt["body_sha256"]
            provenance[str(body)] = sha(body)
        if receipt["status"] not in ["historical_response_received", "date_echo_missing_or_mismatched"]:
            audit_rows.append(row)
            continue
        payload = json.loads(body.read_text())
        status, detail = inspect_payload(payload, job.date)
        assert status == receipt["status"]
        missing_echo = status == "date_echo_missing_or_mismatched" and not detail["response_next_dates"]
        if status != "historical_response_received" and not missing_echo:
            audit_rows.append(row)
            continue
        if len(payload["data"]["plate"]) >= 100:
            row["proof_status"] = "possible_topic_limit"
            audit_rows.append(row)
            continue
        try:
            rr, aa = extract_identities(payload["data"], job.date, job.source_date)
        except (ValueError, TypeError, KeyError):
            row["proof_status"] = "invalid_identity_structure"
            audit_rows.append(row)
            continue
        relations.extend(rr)
        table = pd.DataFrame(aa, columns=["date", "source_date", "code", "source_amount_yi"]).drop_duplicates()
        if len(table):
            table["local_amount"] = [daily_amounts.get((job.source_date, code), np.nan) for code in table.code]
            table["amount_difference"] = table.source_amount_yi * 1e8 - table.local_amount
            table["amount_match"] = (np.isfinite(table.source_amount_yi) & table.source_amount_yi.gt(0)
                                     & table.amount_difference.abs().le(500000.02))
            amounts.extend(table.to_dict("records"))
            row["amount_stocks"] = int(table.code.nunique())
            row["amount_conflicts"] = int(table.code.duplicated().sum())
            row["amount_mismatches"] = int((~table.amount_match).sum())
            finite = table.amount_difference.abs().dropna()
            row["max_amount_difference"] = float(finite.max()) if len(finite) else None
        good = row["amount_stocks"] >= 5 and row["amount_conflicts"] == 0 and row["amount_mismatches"] == 0
        row["source_valid"] = good
        row["proof_status"] = ("amount_and_echo_verified" if not missing_echo else "amount_verified_echo_absent") if good else "amount_date_proof_failed"
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    raw = pd.DataFrame(relations)
    relation = raw[["date", "source_date", "code", "plate_id"]].drop_duplicates()
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.register("states", states)
    c.register("audit", audit)
    c.register("all_relations", relation)
    c.read_parquet(str(ROOT / "visible_main.parquet")).create_view("visible")
    c.execute("""CREATE TABLE valid_relations AS SELECT r.* FROM all_relations r
        JOIN audit a USING(date,source_date) WHERE a.source_valid""")
    c.execute("""CREATE TABLE peers AS SELECT r.*,v.code IS NOT NULL AS visible,
        v.return_1449 FROM valid_relations r JOIN states s USING(date,source_date,code)
        LEFT JOIN visible v USING(date,code) WHERE s.prior_status='closed_limit'""")
    c.execute("""CREATE TABLE theme_context AS SELECT date,source_date,plate_id,count(*) AS members,
        sum(visible::INT) AS visible_members,coalesce(sum(return_1449),0) AS return_sum
        FROM peers GROUP BY date,source_date,plate_id""")
    c.execute("""CREATE TABLE choices AS WITH counted AS (
        SELECT r.*,(s.prior_status='closed_limit') IS TRUE AS self_member,
          coalesce(t.members,0)-((s.prior_status='closed_limit') IS TRUE)::INT AS peer_count,
          coalesce(t.visible_members,0)-((s.prior_status='closed_limit') IS TRUE)::INT AS peer_visible,
          coalesce(t.return_sum,0)-CASE WHEN s.prior_status='closed_limit' THEN v.return_1449 ELSE 0 END AS peer_return_sum
        FROM valid_relations r JOIN visible v USING(date,code)
        LEFT JOIN states s USING(date,source_date,code) LEFT JOIN theme_context t USING(date,source_date,plate_id))
        SELECT * FROM counted QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY peer_count DESC,plate_id)=1""")
    features = c.sql("""WITH market AS (SELECT date,count(*) AS market_count,avg(return_1449) AS market_return
        FROM visible GROUP BY date)
        SELECT v.*,a.source_date,a.source_valid,a.proof_status,coalesce(s.prior_status,'unknown') AS prior_status,
          q.plate_id AS selected_theme,q.self_member,q.peer_count,q.peer_visible,q.peer_return_sum,
          m.market_count,m.market_return
        FROM visible v JOIN audit a USING(date) JOIN market m USING(date)
        LEFT JOIN states s USING(date,code) LEFT JOIN choices q USING(date,code)
        ORDER BY date,code""").df()
    features["context_coverage"] = features.peer_visible / features.peer_count.replace(0, np.nan)
    good = (features.source_valid & features.peer_count.ge(3) & features.context_coverage.ge(.9)).fillna(False)
    features["peer_return"] = (features.peer_return_sum / features.peer_visible.replace(0, np.nan)).where(good)
    features["excess_premium"] = (features.peer_return - features.market_return).where(good)
    features["context"] = np.select([~good, features.excess_premium.ge(.01), features.excess_premium.le(-.01)],
                                    ["unknown", "strong", "weak"], default="neutral")
    features["context_reason"] = np.select([
        ~features.source_valid, features.selected_theme.isna(), features.peer_count.lt(3).fillna(True),
        features.context_coverage.lt(.9).fillna(True)],
        ["source_unknown", "no_reported_theme", "too_few_peers", "peer_visibility_insufficient"], default="known")
    features["primary"] = features.necessary_tradeable & features.prior_status.eq("broken_limit") & features.context.eq("strong")
    assert len(features) == manifest["main_stock_days"] and not features.duplicated(["date", "code"]).any()
    assert features.necessary_tradeable.sum() == manifest["necessary_stock_days"]
    assert features.source_date.lt(features.date).all()
    assert features.loc[features.selected_theme.notna(), "peer_count"].ge(0).all()
    outputs = {"source_audit": audit, "raw_relations": raw, "relations": relation,
               "amount_audit": pd.DataFrame(amounts), "features": features}
    for table in ["peers", "theme_context", "choices"]:
        outputs[table] = c.sql("SELECT * FROM " + table + " ORDER BY date,plate_id").df()
    for name, frame in outputs.items():
        frame.to_parquet(ROOT / (name + ".parquet"), index=False, compression="zstd")
    report = dict(manifest_sha256=sha(ROOT / "manifest.json"), history_report_sha256=sha(ROOT / "history_report.json"),
                  download_report_sha256=sha(ROOT / "download_report.json"), date_proof_sha256=sha(DATE_PROOF),
                  source_sha256=provenance, rows=len(features), primary=int(features.primary.sum()),
                  source_statuses=audit.proof_status.value_counts().to_dict(),
                  primary_by_half=features.groupby("half").primary.sum().to_dict(),
                  necessary_coverage=features.loc[features.necessary_tradeable].groupby(
                      ["half", "prior_status", "context"]).size().rename("rows").reset_index().to_dict("records"),
                  output_sha256={name + ".parquet": sha(ROOT / (name + ".parquet")) for name in outputs},
                  outcomes_read=False, prices_2026_read=False, precise_first_publication_verified=False)
    save_json(ROOT / "input_report.json", report)
    return {k: v for k, v in report.items() if k not in ["source_sha256", "necessary_coverage"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["history", "inputs"])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
