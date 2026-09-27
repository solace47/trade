"""Download only the fixed, outcome-independent historical tick-flow sample."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

import probe_historical_ticks as wire
from trade_research.corporate_cash import save_json, sha

ROOT = Path("data/research/tick_flow_winner")
PROTOCOL = Path("config/tick_flow_winner_protocol.json")


def main():
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not change downloads after input features have been frozen")
    cfg = json.loads(PROTOCOL.read_text())
    frozen = json.loads((ROOT / "cohort_report.json").read_text())
    assert sha(PROTOCOL) == frozen["protocol_sha256"]
    assert sha(ROOT / "cohort.parquet") == frozen["cohort_sha256"]
    cohort = pd.read_parquet(ROOT / "cohort.parquet")
    available = json.loads(Path("data/research/tick_source_probe/availability.json").read_text())
    hosts = [r["host"] for r in available["attempts"] if r["status"] == "nonempty"][:cfg["max_hosts_per_session"]]
    assert len(hosts) == cfg["max_hosts_per_session"]
    network = json.loads(Path("config/tick_source_probe_protocol.json").read_text())
    raw_exchange = wire.exchange
    next_request = 0.0

    def paced_exchange(sock, packet, prefix):
        nonlocal next_request
        wait = next_request - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_request = time.monotonic() + cfg["request_interval_seconds"]
        return raw_exchange(sock, packet, prefix)

    wire.exchange = paced_exchange
    sock, connected_host = None, None
    started = time.monotonic()
    receipts = []
    try:
        for index, row in enumerate(cohort.itertuples(index=False), start=1):
            leaf = ROOT / "sessions" / row.date / row.code.replace(".", "_")
            receipt_path = leaf / "receipt.json"
            prior_attempts = []
            if receipt_path.exists():
                cached = json.loads(receipt_path.read_text())
                if cached["status"] == "downloaded" and cached.get("rows", 0) > 0:
                    assert sha(leaf / "ticks.parquet") == cached["ticks_sha256"]
                    receipts.append(cached)
                    continue
                prior_attempts = cached["attempts"]
            receipt = dict(date=row.date, code=row.code, status="source_unknown", attempts=prior_attempts)
            for host in hosts:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                attempt_path = leaf / "wire" / stamp
                attempt = dict(host=host, wire_folder=str(attempt_path), status="pending")
                receipt["attempts"].append(attempt)
                try:
                    if sock is None or connected_host != host:
                        if sock is not None:
                            sock.close()
                        sock = wire.connect(host, network, attempt_path)
                        connected_host = host
                    rows = []
                    for page in range(cfg["max_pages"]):
                        part = wire.request_history(sock, row.code, row.date,
                            page * cfg["page_size"], cfg["page_size"], attempt_path / f"history_{page}")
                        rows = part + rows
                        if len(part) < cfg["page_size"]:
                            break
                    else:
                        raise ValueError("earliest session boundary was not reached")
                    if not rows:
                        # These fixed names traded that day in the independent
                        # source. A valid empty response is missing history,
                        # so try the other already-verified server once.
                        raise ValueError("empty historical session for fixed active stock-day")
                    if any(b["minute"] < a["minute"] for a, b in zip(rows, rows[1:])):
                        raise ValueError("nonchronological historical records")
                    frame = pd.DataFrame(rows, columns=["minute", "price_raw", "volume_raw", "direction_raw", "reserved_raw"])
                    for name in frame:
                        frame[name] = frame[name].astype("int64")
                    frame["tick_seq"] = range(len(frame))
                    leaf.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(leaf / "ticks.parquet", index=False, compression="zstd")
                    attempt.update(status="downloaded", pages=page + 1, rows=len(rows))
                    receipt.update(status="downloaded", ticks_sha256=sha(leaf / "ticks.parquet"),
                        rows=len(rows), request_response_sha256={str(p):sha(p) for p in sorted(attempt_path.glob("*.bin"))})
                    break
                except Exception as exc:
                    attempt.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    if sock is not None:
                        sock.close()
                    sock, connected_host = None, None
            save_json(receipt_path, receipt)
            receipts.append(receipt)
            if index % 25 == 0 or index == len(cohort):
                progress = dict(completed=index, total=len(cohort),
                    downloaded=sum(r["status"] == "downloaded" for r in receipts),
                    source_unknown=sum(r["status"] != "downloaded" for r in receipts),
                    elapsed_seconds=round(time.monotonic() - started, 1), last_date=row.date)
                save_json(ROOT / "download_progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    finally:
        if sock is not None:
            sock.close()
    summary = dict(protocol_sha256=sha(PROTOCOL), cohort_report_sha256=sha(ROOT / "cohort_report.json"),
        sessions=len(receipts), rows=sum(r.get("rows", 0) for r in receipts),
        status_counts=pd.Series([r["status"] for r in receipts]).value_counts().to_dict(),
        prices_2026_read=False, outcomes_read=False,
        receipt_sha256={str(ROOT / "sessions" / r["date"] / r["code"].replace(".", "_") / "receipt.json"):
            sha(ROOT / "sessions" / r["date"] / r["code"].replace(".", "_") / "receipt.json") for r in receipts})
    save_json(ROOT / "download_report.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "receipt_sha256"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
