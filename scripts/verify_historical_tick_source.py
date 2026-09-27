"""Replay every archived wire page independently and compare fixed-day prices.

No strategy outcomes are loaded. Tick prices are representative prices, so
their extrema need not attain exchange daily extrema. Lots are rounded, and
price * volume is explicitly an approximate turnover, not an exchange amount.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zlib

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/tick_source_probe"


def independent_decode(raw):
    compressed, expanded = struct.unpack_from("<HH", raw, 12)
    assert len(raw) == 16 + compressed
    payload = raw[16:] if compressed == expanded else zlib.decompress(raw[16:])
    assert len(payload) == expanded and expanded >= 6
    count = int.from_bytes(payload[:2], "little")
    cursor, accumulated, result = 6, 0, []
    for _ in range(count):
        minute = int.from_bytes(payload[cursor:cursor + 2], "little")
        cursor += 2
        values = []
        for _ in range(4):
            bytes_ = [payload[cursor]]
            cursor += 1
            while bytes_[-1] >= 128:
                bytes_.append(payload[cursor])
                cursor += 1
            magnitude = (bytes_[0] % 64) + sum(
                (value % 128) * 2 ** (6 + 7 * index)
                for index, value in enumerate(bytes_[1:]))
            values.append(-magnitude if bytes_[0] & 64 else magnitude)
        accumulated += values[0]
        result.append(dict(minute=minute, price_raw=accumulated,
                           volume_raw=values[1], direction_raw=values[2],
                           reserved_raw=values[3]))
    assert cursor == len(payload)
    return result


def main():
    manifest = json.loads((DATA / "sessions.json").read_text())
    cfg_bytes = (ROOT / "config/tick_source_probe_protocol.json").read_bytes()
    cfg = json.loads(cfg_bytes)
    assert manifest["config_sha256"] == hashlib.sha256(cfg_bytes).hexdigest()
    attempt = manifest["attempts"][0]
    assert attempt["status"] == "downloaded_pending_validation"
    expected = {(s, d) for s in cfg["sample_symbols"] for d in cfg["sample_dates"]}
    assert {(x["symbol"], x["date"]) for x in attempt["sessions"]} == expected
    con = duckdb.connect()
    audited, pages, fingerprints = [], 0, []
    for item in attempt["sessions"]:
        path = ROOT / item["path"]
        original = json.loads(path.read_text())
        ex, code = item["symbol"].split(".")
        date = item["date"]
        replay = []
        for page in range(cfg["max_session_pages"]):
            prefix = path.parent / f"{item['symbol']}_{date}_{page}"
            request = prefix.with_name(prefix.name + ".request.bin").read_bytes()
            raw_path = prefix.with_name(prefix.name + ".response.bin")
            raw = raw_path.read_bytes()
            assert request[:12] == bytes.fromhex("0c013001000112001200b50f")
            requested_date, market, encoded_code, start, count = struct.unpack("<IH6sHH", request[12:])
            assert requested_date == int(date.replace("-", ""))
            assert market == int(ex == "sh") and encoded_code.decode() == code
            assert start == page * cfg["session_page_size"] and count == cfg["session_page_size"]
            decoded = independent_decode(raw)
            replay = decoded + replay
            pages += 1
            fingerprints.append(dict(path=str(raw_path.relative_to(ROOT)),
                                     sha256=hashlib.sha256(raw).hexdigest()))
            if len(decoded) < count:
                break
        else:
            raise AssertionError("session's earliest boundary is missing")
        assert original == replay and len(replay) == item["rows"]
        minute = np.array([r["minute"] for r in replay])
        price = np.array([r["price_raw"] for r in replay]) / 100
        volume = np.array([r["volume_raw"] for r in replay]) * 100
        direction = np.array([r["direction_raw"] for r in replay])
        assert np.all(np.diff(minute) >= 0)
        valid = (minute == 565) | ((minute >= 570) & (minute <= 690)) | ((minute >= 780) & (minute <= 900)) | ((minute >= 905) & (minute <= 930))
        assert valid.all() and minute[0] == 565
        regular = minute <= 900
        assert (direction[~regular] == 5).all() and (direction[regular] != 5).all()
        assert np.isin(direction, [0, 1, 2, 5]).all()
        daily = con.execute("select * from read_parquet(?) where date=?", [
            str(ROOT / f"data/baostock/market_2020_2026/daily/{ex}_{code}.parquet"), date]).fetchdf()
        assert len(daily) == 1
        d = daily.iloc[0]
        assert d["adjustflag"] == 3 and d["tradestatus"] == 1
        assert (price[regular] >= d["low"] - 0.000001).all()
        assert (price[regular] <= d["high"] + 0.000001).all()
        assert abs(price[regular][0] - d["open"]) < 0.000001
        assert abs(price[regular][-1] - d["close"]) < 0.000001
        minute_frame = con.execute("select timestamp,volume from read_parquet(?) where cast(timestamp as date)=cast(? as date) order by timestamp", [
            str(ROOT / f"data/hf/pilot/data/stock_1m/{ex.upper()}/{code}.parquet"), date]).fetchdf()
        assert len(minute_frame) == 241
        minute_frame["clock"] = minute_frame.timestamp.dt.hour * 60 + minute_frame.timestamp.dt.minute
        prefix_differences = []
        for checkpoint in (600, 660, 840, 868):
            tick_sum = int(volume[minute <= checkpoint].sum())
            for offset in (0, 1):
                minute_sum = int(minute_frame.loc[minute_frame.clock <= checkpoint + offset, "volume"].sum())
                prefix_differences.append(dict(tick_clock=checkpoint, minute_label_offset=offset,
                    tick_volume=tick_sum, minute_volume=minute_sum,
                    difference_shares=tick_sum-minute_sum))
        move = np.sign(np.diff(price[regular]))
        inferred = direction[regular][1:]
        directional = ((inferred == 0) | (inferred == 1)) & (move != 0)
        matches = ((move > 0) & (inferred == 0)) | ((move < 0) & (inferred == 1))
        audited.append(dict(symbol=item["symbol"], date=date, rows=len(replay),
            regular_rows=int(regular.sum()), after_hours_rows=int((~regular).sum()),
            volume_difference_shares=int(volume[regular].sum() - d["volume"]),
            approximate_amount_difference_bps=float((np.sum(price[regular] * volume[regular]) / d["amount"] - 1) * 10000),
            representative_high_difference=float(price[regular].max() - d["high"]),
            representative_low_difference=float(price[regular].min() - d["low"]),
            direction_matches_nonflat_price_move=float(matches[directional].mean()),
            flat_price_record_fraction=float((move == 0).mean()),
            prefix_volume_comparisons=prefix_differences))
    report = dict(status="fixed_sample_source_reconciled", config_sha256=manifest["config_sha256"],
        sessions=len(audited), records=sum(r["rows"] for r in audited), wire_pages=pages,
        max_abs_volume_difference_shares=max(abs(r["volume_difference_shares"]) for r in audited),
        max_abs_approx_amount_difference_bps=max(abs(r["approximate_amount_difference_bps"]) for r in audited),
        evaluation_outcomes_read=False, prices_2026_read=False,
        limitation="固定12日证实可获取与日级量价相符；不是全历史质量证明。方向无独立交易所真值；代表价可能不含日内极值；以手汇总有取整，价乘量仅近似额。含14:49分钟的分笔可能发生到14:49:59，新研究须只用至14:48分钟，避免读取14:49之后。", 
        details=audited, wire_sha256=fingerprints)
    target = DATA / "validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"details", "wire_sha256"}},ensure_ascii=False))
    print("sha256", hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
