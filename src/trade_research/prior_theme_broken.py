"""Freeze preceding-session theme jobs and cache anonymous historical responses."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

import duckdb
import pandas as pd

from .corporate_cash import save_json, sha
from .network import effective_environment

ROOT = Path("data/research/prior_theme_broken")
PROTOCOL = Path("config/prior_theme_broken_protocol.json")
BASE = Path("data/research/next_day_winner/visible_base.parquet")
CALENDAR = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")
PILOT = Path("data/research/theme_source_probe/zzshare/curl_initial")
URL = "https://api.zizizaizai.com/v3/open/review/uplimit/hot"


def freeze():
    if (ROOT / "manifest.json").exists():
        raise ValueError("Do not replace the new theme protocol or dates")
    ROOT.mkdir(parents=True, exist_ok=True)
    original = json.loads(BASE.with_name("base_report.json").read_text())
    assert sha(BASE) == original["base_sha256"]
    calendar = pd.read_parquet(CALENDAR)
    days = sorted(calendar.loc[calendar.is_trading_day.eq("1") & calendar.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    c = duckdb.connect()
    eligible = c.execute("""select distinct date from read_parquet(?) where board='main' and necessary_tradeable
        and date between '2024-01-03' and '2025-12-30' order by date""", [str(BASE)]).df()
    previous = dict(zip(days[1:], days[:-1]))
    jobs = eligible.assign(source_date=eligible.date.map(previous))
    assert jobs.source_date.notna().all() and jobs.source_date.lt(jobs.date).all()
    assert jobs.source_date.between("2024-01-02", "2025-12-29").all()
    jobs.to_parquet(ROOT / "source_jobs.parquet", index=False, compression="zstd")
    main = c.execute("""select date,code,half,board,necessary_tradeable,decision_shares,price_1449,preclose,
        return_1449,return_tail29,amount_1449,return20_prior_adjusted,isST,tradestatus,listing_age_sessions
        from read_parquet(?) where board='main' and date between '2024-01-03' and '2025-12-30' order by date,code""", [str(BASE)]).df()
    main.to_parquet(ROOT / "visible_main.parquet", index=False, compression="zstd")
    result = dict(protocol_sha256=sha(PROTOCOL), base_sha256=sha(BASE), calendar_sha256=sha(CALENDAR),
                  jobs_sha256=sha(ROOT / "source_jobs.parquet"), main_sha256=sha(ROOT / "visible_main.parquet"),
                  source_audit_sha256=sha(Path("data/research/theme_source_probe/audit/report.json")),
                  source_dates=len(jobs), first_source=jobs.source_date.min(), last_source=jobs.source_date.max(),
                  main_stock_days=len(main), necessary_stock_days=int(main.necessary_tradeable.sum()),
                  outcomes_read=False, requested_2026_prices=False)
    save_json(ROOT / "manifest.json", result)
    return result


def inspect_payload(payload, signal_date):
    if payload.get("code") not in [200, 20000]:
        return "business_error", {}
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("today") is not False:
        return "invalid_historical_structure", {}
    if not all(isinstance(data.get(key), dict) for key in ["plate_stocks", "plate_stocks_zb", "plate_stocks_bx"]):
        return "missing_membership_maps", {}
    if not isinstance(data.get("plate"), list) or not data["plate"]:
        return "empty_historical_payload", {}
    # Only next-session dates are checked as response provenance. The response's
    # next-day price fields are never read, copied or used as predictors.
    next_dates = sorted({str(row["next_day"]) for rows in data["plate_stocks"].values()
                         for row in rows if row.get("next_day")})
    if next_dates != [signal_date]:
        return "date_echo_missing_or_mismatched", dict(response_next_dates=next_dates)
    if len(data["plate"]) >= 100:
        return "possible_topic_limit", dict(themes=len(data["plate"]))
    fields = dict(themes=len(data["plate"]), source_stocks=len(data.get("stock_info", {})),
                  relationships=sum(len(v) for name in ["plate_stocks", "plate_stocks_zb", "plate_stocks_bx"] for v in data[name].values()))
    return "historical_response_received", fields


def fetch():
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not replace source after inputs are frozen")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["protocol_sha256"] == sha(PROTOCOL)
    assert manifest["jobs_sha256"] == sha(ROOT / "source_jobs.parquet")
    jobs = pd.read_parquet(ROOT / "source_jobs.parquet")
    cache = ROOT / "source"
    cache.mkdir(exist_ok=True)
    receipts = []
    stopped = None
    for job in jobs.itertuples(index=False):
        body = cache / (job.source_date + ".body")
        meta = cache / (job.source_date + ".json")
        date1 = job.source_date.replace("-", "")
        url = URL + "?" + urlencode(dict(date1=date1, limit=100))
        if meta.exists():
            receipt = json.loads(meta.read_text())
            assert receipt["url"] == url and receipt["signal_date"] == job.date
            if receipt.get("body_sha256"):
                assert sha(body) == receipt["body_sha256"]
        else:
            pilot = PILOT / (date1 + "_hot.body")
            reused = pilot.exists()
            if reused:
                proof = json.loads((PILOT / "report.json").read_text())
                match = next(p for p in proof["responses"] if p["date"] == date1 and p["kind"] == "hot")
                assert sha(pilot) == match["body_sha256"]
                body.write_bytes(pilot.read_bytes())
                exit_code, http_code, error = 0, "200", ""
            else:
                r = subprocess.run(["curl", "--silent", "--show-error", "--max-time", "25", "--connect-timeout", "10",
                                    "--header", "sdk-key: anonymous", "--user-agent", "trade-research/1.0",
                                    "--output", str(body), "--write-out", "%{http_code}", url],
                                   env=effective_environment(), capture_output=True, text=True, timeout=30)
                exit_code, http_code, error = r.returncode, r.stdout, r.stderr
            receipt = dict(source_date=job.source_date, signal_date=job.date, url=url, curl_exit=exit_code,
                           http_code=http_code, error=error, reused_pilot=reused, body_sha256=sha(body) if body.exists() else None)
            if exit_code or http_code != "200":
                receipt["status"] = "transport_or_http_error"
            else:
                try:
                    receipt["status"], details = inspect_payload(json.loads(body.read_text()), job.date)
                    receipt.update(details)
                except (ValueError, TypeError, KeyError):
                    receipt["status"] = "invalid_json_or_structure"
            save_json(meta, receipt)
            if not reused:
                time.sleep(3)
        receipts.append(receipt)
        if len(receipts) % 20 == 0 or receipt["status"] != "historical_response_received":
            print(json.dumps(dict(completed=len(receipts), total=len(jobs), source_date=job.source_date,
                                  status=receipt["status"]), ensure_ascii=False), flush=True)
        if receipt["http_code"] in ["401", "403", "429"] or receipt["status"] == "business_error":
            stopped = receipt["status"]
            break
    result = dict(manifest_sha256=sha(ROOT / "manifest.json"), requested_dates=len(jobs), completed_dates=len(receipts),
                  all_dates_attempted=len(receipts) == len(jobs), stopped_on=stopped,
                  statuses=pd.Series([r["status"] for r in receipts]).value_counts().to_dict(),
                  receipts_sha256={r["source_date"]: sha(cache / (r["source_date"] + ".json")) for r in receipts},
                  outcomes_read=False, requested_2026_prices=False)
    save_json(ROOT / "download_report.json", result)
    return {k: v for k, v in result.items() if k != "receipts_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "fetch"])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
