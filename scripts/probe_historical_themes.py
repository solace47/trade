"""Read-only anonymous historical-source probe, without importing the SDK.

Request field meanings from fleetinglife/levistock at
09f871a77b6e3241b1249d986e7e8a7cbf9a44fe (MIT); license retained in docs/licenses.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from urllib.parse import urlencode

import requests

from trade_research.corporate_cash import save_json, sha
from trade_research.network import effective_environment

ROOT = Path("data/research/theme_source_probe")
PROTOCOL = Path("config/theme_source_probe_protocol.json")
URL = "https://apphis.kaipanhong.com/w1/api/index.php"


def run(transport="requests"):
    report_file = ROOT / ("probe_report.json" if transport == "requests" else "curl_probe_report.json")
    if report_file.exists():
        raise ValueError("Preserve inspected probe responses")
    config = json.loads(PROTOCOL.read_text())
    manifest = ROOT / "probe_manifest.json"
    if not manifest.exists():
        save_json(manifest, dict(protocol_sha256=sha(PROTOCOL), device_id=str(uuid.uuid4()),
                                 source_manifest_sha256=sha(ROOT / "source_manifest.json")))
    identity = json.loads(manifest.read_text())
    assert identity["protocol_sha256"] == sha(PROTOCOL)
    base = dict(PhoneOSNew="1", DeviceID=identity["device_id"], VerSion="6.0.6",
                Token="0", UserID="0", Red="1", apiv="w45")
    templates = {
        "limit_up": dict(a="HisDaBanList", c="HisHomeDingPan", Order="1", st="50", Index="0", Is_st="1",
                         PidType="4", Type="6", FilterMotherboard="0", Filter="0", FilterTIB="0", FilterGem="0"),
        "themes": dict(a="RealRankingInfo", c="ZhiShuRanking", Order="1", st="50", Index="0", Type="1", ZSType="7"),
        "resumption": dict(a="GetPlateInfo_w38", c="HisLimitResumption", st="100", Index="0"),
        "events": dict(a="GetPMSL_PMLD", c="FuPanLa", st="100", Index="0"),
    }
    os.environ.update(effective_environment())
    session = requests.Session()
    session.headers.update({"User-Agent": "trade-research/1.0", "Accept": "application/json"})
    if transport == "curl":
        def curl_post(url, data, timeout):
            temporary = ROOT / "curl_pending.body"
            result = subprocess.run(["curl", "--silent", "--show-error", "--max-time", "25", "--connect-timeout", "10",
                                     "--header", "Content-Type: application/x-www-form-urlencoded", "--user-agent", "trade-research/1.0",
                                     "--data-binary", "@-", "--output", str(temporary), "--write-out", "%{http_code}", url],
                                    input=urlencode(data), env=effective_environment(), capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise requests.ConnectionError("System curl: " + result.stderr)
            response = requests.Response()
            response.status_code = int(result.stdout)
            response._content = temporary.read_bytes()
            response.headers["Content-Type"] = "unverified; retained body"
            temporary.unlink()
            return response
        session.post = curl_post
    outcomes = []
    stop = None
    folder = ROOT / ("initial" if transport == "requests" else "curl_initial")
    folder.mkdir(exist_ok=True)
    for day in config["dates"]:
        for kind, fields in templates.items():
            prefix = folder / (day + "_" + kind)
            params = dict(base, **fields)
            params["Day" if kind == "limit_up" else "Date"] = day
            if prefix.with_suffix(".receipt.json").exists():
                receipt = json.loads(prefix.with_suffix(".receipt.json").read_text())
                assert receipt["params"] == params
                if "response_sha256" in receipt:
                    assert sha(prefix.with_suffix(".body")) == receipt["response_sha256"]
            else:
                receipt = dict(date=day, kind=kind, url=URL, params=params, status="unavailable")
                try:
                    response = session.post(URL, data=params, timeout=(10, 20))
                    prefix.with_suffix(".body").write_bytes(response.content)
                    receipt.update(http_status=response.status_code, response_sha256=sha(prefix.with_suffix(".body")),
                                   content_type=response.headers.get("Content-Type"), bytes=len(response.content))
                    if response.status_code in [401, 403, 429]:
                        receipt["status"] = "access_or_rate_rejected"
                    elif response.status_code != 200:
                        receipt["status"] = "http_error"
                    else:
                        try:
                            payload = response.json()
                            receipt["top_level_keys"] = list(payload) if isinstance(payload, dict) else None
                            receipt["error_code"] = payload.get("errcode") if isinstance(payload, dict) else None
                            if isinstance(payload, dict) and str(payload.get("errcode")) == "0":
                                receipt["status"] = "response_received_not_yet_verified"
                                receipt["list_counts"] = {k: len(v) for k, v in payload.items() if isinstance(v, list)}
                            else:
                                receipt["status"] = "nonzero_or_missing_error_code"
                        except ValueError:
                            receipt["status"] = "non_json"
                except requests.RequestException as error:
                    receipt.update(status="transport_failed", error_type=type(error).__name__, error=str(error))
                save_json(prefix.with_suffix(".receipt.json"), receipt)
                time.sleep(1)
            outcomes.append({k: v for k, v in receipt.items() if k not in ["params", "url", "error"]})
            print(json.dumps(outcomes[-1], ensure_ascii=False), flush=True)
            if receipt["status"] in ["access_or_rate_rejected", "nonzero_or_missing_error_code", "transport_failed", "non_json"]:
                stop = receipt["status"]
                break
        if stop:
            break
    result = dict(protocol_sha256=sha(PROTOCOL), probe_manifest_sha256=sha(manifest), transport=transport,
                  responses=outcomes, stopped_on=stop, historical_source_verified=False,
                  new_2026_prices_read=False, outcomes_used=False)
    save_json(report_file, result)
    return dict(requests=len(outcomes), stopped_on=stop, historical_source_verified=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["requests", "curl"], default="requests")
    print(json.dumps(run(parser.parse_args().transport), ensure_ascii=False, indent=2))
