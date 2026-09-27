"""Bounded, read-only historical TDX source probe; no current quotes/outcomes.

Wire facts checked against CNEquity 1650e384 / tdxpy 0.2.7.
This is a standalone decoder, not an import of either external project.
See docs/licenses/tdxpy.txt for the original protocol implementation's notice.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import struct
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/tick_source_probe_protocol.json"
OUT = ROOT / "data/research/tick_source_probe"
SETUP = (
    "0c0218930001030003000d0001",
    "0c0218940001030003000d0002",
    "0c031899000120002000db0fd5d0c9ccd6a4a8af0000008fc22540130000d500c9ccbdf0d7ea00000002",
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_exact(sock, size, captured):
    chunk = bytearray()
    while len(chunk) < size:
        part = sock.recv(size - len(chunk))
        if not part:
            raise EOFError(f"connection closed after {len(chunk)} of {size} bytes")
        chunk.extend(part)
        captured.extend(part)
    return bytes(chunk)


def exchange(sock, packet, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + ".request.bin").write_bytes(packet)
    captured = bytearray()
    try:
        sock.sendall(packet)
        header = read_exact(sock, 16, captured)
        _, _, _, wire_size, expanded_size = struct.unpack("<IIIHH", header)
        body = read_exact(sock, wire_size, captured)
        if wire_size != expanded_size:
            decoder = zlib.decompressobj()
            body = decoder.decompress(body, expanded_size + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("invalid or incomplete zlib response")
        if len(body) != expanded_size:
            raise ValueError("response length differs from header")
        return body
    finally:
        prefix.with_name(prefix.name + ".response.bin").write_bytes(captured)


def decode_history(body, requested):
    # An unavailable session (or the page beyond an exact multiple) may carry
    # just the zero count, without the four normal filler bytes.
    if body == b"\x00\x00":
        return []
    if len(body) < 6:
        raise ValueError("history body shorter than count plus filler")
    count = struct.unpack_from("<H", body)[0]
    if count > requested:
        raise ValueError("response count exceeds requested bound")
    pos = 6

    def signed_integer():
        nonlocal pos
        first = body[pos]
        pos += 1
        value = first & 63
        negative = bool(first & 64)
        more, shift = first & 128, 6
        while more:
            if shift > 55:
                raise ValueError("unbounded integer encoding")
            byte = body[pos]
            pos += 1
            value |= (byte & 127) << shift
            more, shift = byte & 128, shift + 7
        return -value if negative else value

    price, rows = 0, []
    for _ in range(count):
        minute = struct.unpack_from("<H", body, pos)[0]
        pos += 2
        price += signed_integer()
        volume, direction, reserved = (signed_integer() for _ in range(3))
        if not 0 <= minute < 1440 or price <= 0 or volume < 0:
            raise ValueError("invalid time/price/volume")
        rows.append(dict(minute=minute, price_raw=price, volume_raw=volume,
                         direction_raw=direction, reserved_raw=reserved))
    if pos != len(body):
        raise ValueError(f"unparsed history bytes: {len(body) - pos}")
    return rows


def request_history(sock, symbol, date, start, count, prefix):
    assert date[:4] in {"2024", "2025"}
    market, code = symbol.split(".")
    assert market in {"sh", "sz"} and len(code) == 6 and code.isdigit()
    packet = bytes.fromhex("0c013001000112001200b50f") + struct.pack(
        "<IH6sHH", int(date.replace("-", "")), int(market == "sh"),
        code.encode("ascii"), start, count)
    body = exchange(sock, packet, prefix)
    return decode_history(body, count)


def connect(host, config, folder):
    sock = socket.create_connection((host, config["port"]), config["timeout_seconds"])
    try:
        for i, packet in enumerate(SETUP):
            exchange(sock, bytes.fromhex(packet), folder / f"setup_{i}")
        return sock
    except Exception:
        sock.close()
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", action="store_true")
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = OUT / "wire" / stamp
    report = dict(config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  started_utc=stamp, attempts=[], read_outcomes=False, read_2026_prices=False)
    available = []
    if args.sessions:
        previous = json.loads((OUT / "availability.json").read_text())
        assert previous["config_sha256"] == report["config_sha256"]
        available = [r["host"] for r in previous["attempts"] if r["status"] == "nonempty"]
        hosts = available[:1]
        if not hosts:
            raise RuntimeError("no verified nonempty historical endpoint")
    else:
        hosts = config["hosts"]
    for host in hosts:
        started = time.monotonic()
        attempt = dict(host=host, status="unknown", rows=None)
        report["attempts"].append(attempt)
        try:
            with connect(host, config, folder / host) as sock:
                if not args.sessions:
                    rows = request_history(sock, config["availability_symbol"],
                        config["availability_date"], 0, config["availability_rows"],
                        folder / host / "availability")
                    attempt.update(status="nonempty" if rows else "empty", rows=len(rows))
                    save_json(folder / host / "availability_rows.json", rows)
                else:
                    attempt["sessions"] = []
                    for symbol in config["sample_symbols"]:
                        for date in config["sample_dates"]:
                            rows = []
                            size = config["session_page_size"]
                            for page in range(config["max_session_pages"]):
                                part = request_history(sock, symbol, date, page * size, size,
                                    folder / host / f"{symbol}_{date}_{page}")
                                rows = part + rows
                                if len(part) < size:
                                    break
                            else:
                                raise ValueError("full session boundary was not reached")
                            if any(b["minute"] < a["minute"] for a, b in zip(rows, rows[1:])):
                                raise ValueError("session is not chronological")
                            path = folder / host / f"{symbol}_{date}.json"
                            save_json(path, rows)
                            attempt["sessions"].append(dict(symbol=symbol, date=date,
                                rows=len(rows), path=str(path.relative_to(ROOT))))
                    attempt.update(status="downloaded_pending_validation",
                                   rows=sum(r["rows"] for r in attempt["sessions"]))
        except Exception as exc:
            attempt.update(status="error", error_type=type(exc).__name__, error=str(exc))
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)
        report["wire_folder"] = str(folder.relative_to(ROOT))
        save_json(folder / "report.json", report)
        save_json(OUT / ("sessions.json" if args.sessions else "availability.json"), report)


if __name__ == "__main__":
    main()
