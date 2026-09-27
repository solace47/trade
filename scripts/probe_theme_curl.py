"""Preserve four fixed dates using the working system TLS client, anonymously."""
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

from trade_research.corporate_cash import save_json, sha
from trade_research.network import effective_environment

ROOT = Path("data/research/theme_source_probe/zzshare")
PROTOCOL = Path("config/theme_zzshare_probe_protocol.json")


def run():
    dest = ROOT / "curl_initial"
    dest.mkdir(exist_ok=True)
    if (dest / "report.json").exists():
        raise ValueError("Do not replace completed source probe")
    cfg = json.loads(PROTOCOL.read_text())
    outcomes = []
    stop = None
    for day in cfg["dates"]:
        for endpoint in cfg["endpoints"]:
            kind = endpoint.rsplit("/", 1)[-1]
            params = {"date1": day}
            if kind == "hot":
                params["limit"] = 100
            url = endpoint + "?" + urlencode(params)
            file = dest / (day + "_" + kind + ".body")
            meta = file.with_suffix(".receipt.json")
            if meta.exists():
                receipt = json.loads(meta.read_text())
                assert receipt["url"] == url and receipt["body_sha256"] == sha(file)
            else:
                if day == "20240102" and kind == "reason":
                    original = ROOT / "curl_transport_20240102.body"
                    proof = json.loads((ROOT / "curl_transport_report.json").read_text())
                    assert proof["exit_code"] == 0 and proof["body_sha256"] == sha(original)
                    file.write_bytes(original.read_bytes())
                    exit_code, http_code, error = 0, proof["http_code"], ""
                else:
                    r = subprocess.run(["curl", "--silent", "--show-error", "--max-time", "25", "--connect-timeout", "10",
                                        "--header", "sdk-key: anonymous", "--user-agent", "trade-research/1.0",
                                        "--output", str(file), "--write-out", "%{http_code}", url],
                                       env=effective_environment(), capture_output=True, text=True, timeout=30)
                    exit_code, http_code, error = r.returncode, r.stdout, r.stderr
                receipt = dict(date=day, kind=kind, url=url, curl_exit=exit_code, http_code=http_code, error=error,
                               body_sha256=sha(file) if file.exists() else None)
                save_json(meta, receipt)
                time.sleep(3)
            if receipt["curl_exit"] or receipt["http_code"] != "200":
                stop = "transport_or_http_error"
            else:
                payload = json.loads(file.read_text())
                receipt["business_code"] = payload.get("code")
                if payload.get("code") not in [200, 20000]:
                    stop = "business_error"
                else:
                    data = payload.get("data")
                    receipt["data_type"] = type(data).__name__
                    receipt["data_count"] = len(data) if isinstance(data, (list, dict)) else None
                    receipt["keys"] = list(data) if isinstance(data, dict) else None
                    if kind == "reason" and isinstance(data, list):
                        dates = sorted(set(str(row.get("date")) for row in data))
                        receipt["record_dates"] = dates
                        requested = day[:4] + "-" + day[4:6] + "-" + day[6:]
                        if dates and dates != [requested]:
                            stop = "unexpected_record_date"
            outcomes.append(receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k not in ["url", "error"]}, ensure_ascii=False), flush=True)
            if stop:
                break
        if stop:
            break
    report = dict(protocol_sha256=sha(PROTOCOL), source_manifest_sha256=sha(ROOT / "manifest.json"),
                  responses=outcomes, stopped_on=stop, point_in_time_verified=False, economic_outcomes_read=False,
                  dates=cfg["dates"], original_python_failure_preserved=True)
    save_json(dest / "report.json", report)
    return dict(responses=len(outcomes), stopped_on=stop, point_in_time_verified=False)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
