"""Complete the fixed source-date pages and a deterministic member example."""
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

from trade_research.corporate_cash import save_json, sha
from trade_research.network import effective_environment

ROOT = Path("data/research/theme_source_probe")
PROTOCOL = Path("config/theme_source_followup_50_protocol.json")


def run():
    folder = ROOT / "complete_50"
    folder.mkdir(exist_ok=True)
    if (folder / "report.json").exists():
        raise ValueError("Preserve inspected source pagination")
    cfg = json.loads(PROTOCOL.read_text())
    original = json.loads((ROOT / "initial/2024-01-02_limit_up.receipt.json").read_text())
    base_keys = ["PhoneOSNew", "DeviceID", "VerSion", "Token", "UserID", "Red", "apiv"]
    base = {k: original["params"][k] for k in base_keys}
    url = original["url"]
    records = []

    def request(day, kind, fields, offset, page_size):
        name = day + "_" + kind + "_" + str(offset)
        body = folder / (name + ".body")
        receipt_path = folder / (name + ".receipt.json")
        params = dict(base, **fields, st=str(page_size), Index=str(offset))
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["params"] == params and receipt["protocol_sha256"] == sha(PROTOCOL)
            assert receipt["body_sha256"] == sha(body)
        else:
            r = subprocess.run(["curl", "--silent", "--show-error", "--max-time", "25", "--connect-timeout", "10",
                                "--header", "Content-Type: application/x-www-form-urlencoded", "--user-agent", "trade-research/1.0",
                                "--data-binary", "@-", "--output", str(body), "--write-out", "%{http_code}", url],
                               input=urlencode(params), env=effective_environment(), capture_output=True, text=True, timeout=30)
            receipt = dict(date=day, kind=kind, url=url, params=params, curl_exit=r.returncode, http_code=r.stdout,
                           error=r.stderr, body_sha256=sha(body) if body.exists() else None, protocol_sha256=sha(PROTOCOL))
            save_json(receipt_path, receipt)
            time.sleep(1)
        assert receipt["curl_exit"] == 0 and receipt["http_code"] == "200", receipt["error"]
        payload = json.loads(body.read_text())
        assert str(payload.get("errcode")) == "0", (day, kind, payload.get("errcode"))
        assert isinstance(payload.get("list"), list)
        echo = payload.get("Day", payload.get("day", payload.get("date")))
        assert echo in [day, [day]], (day, kind, "wrong response date", echo)
        if offset == 0 and kind in ["themes", "limit_up"]:
            initial = json.loads((ROOT / "curl_initial" / (day + "_" + kind + ".body")).read_text())
            assert [r[0] for r in payload["list"]] == [r[0] for r in initial["list"]], (day, kind, "changed initial page")
        records.append(dict(receipt=receipt_path.name, receipt_sha256=sha(receipt_path),
                            body=body.name, body_sha256=sha(body), rows=len(payload["list"])))
        print(json.dumps(dict(date=day, kind=kind, offset=offset, rows=len(payload["list"])), ensure_ascii=False), flush=True)
        return payload["list"]

    def pages(day, kind, fields):
        all_rows = []
        page_size = 200 if kind.startswith("members_") else 50
        max_pages = 20 if page_size == 200 else 40
        for offset in range(0, page_size * max_pages, page_size):
            chunk = request(day, kind, fields, offset, page_size)
            assert len(chunk) <= page_size
            all_rows.extend(chunk)
            if len(chunk) < page_size:
                codes = [str(x[0]) for x in all_rows]
                assert len(codes) == len(set(codes)), (day, kind, "duplicate page identifiers")
                return all_rows
        raise ValueError("Pagination cap reached; source remains incomplete")

    counts = []
    for day in cfg["dates"]:
        limits = pages(day, "limit_up", dict(a="HisDaBanList", c="HisHomeDingPan", Order="1", Is_st="1", PidType="4",
                       Type="6", FilterMotherboard="0", Filter="0", FilterTIB="0", FilterGem="0", Day=day))
        themes = pages(day, "themes", dict(a="RealRankingInfo", c="ZhiShuRanking", Order="1", Date=day, Type="1", ZSType="7"))
        members = []
        chosen = None
        if themes:
            chosen = min(themes, key=lambda x: str(x[0]))
            members = pages(day, "members_" + str(chosen[0]), dict(a="ZhiShuStockList_W8", c="ZhiShuRanking", Order="1",
                            old="1", Date=day, Type="6", PlateID=str(chosen[0]), IsZZ="0", IsKZZType="0", TSZB="0", TSZB_Type="0", filterType="0"))
        counts.append(dict(date=day, limit_stocks=len(limits), themes=len(themes),
                           selected_plate=chosen[:2] if chosen else None, members=len(members),
                           membership_status="response_received_unverified" if themes else "theme_source_unavailable"))
    result = dict(protocol_sha256=sha(PROTOCOL), requests=records, counts=counts,
                  economic_outcomes_read=False, point_in_time_verified=False)
    save_json(folder / "report.json", result)
    return counts


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
