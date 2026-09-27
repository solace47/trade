"""Independent clock-string windows and scalar Decimal execution accounting."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.corporate_cash import MINUTES, save_json, sha

ROOT = Path("data/research/tick_morning_exit")


def windows(ROOT=ROOT):
    report = json.loads((ROOT / "raw_report.json").read_text())
    for path, digest in report["parts_sha256"].items():
        assert sha(Path(path)) == digest
    raw = pd.concat([pd.read_parquet(p) for p in report["parts_sha256"]], ignore_index=True)
    assert len(raw) == report["raw_rows"]
    assert raw.timestamp.dt.strftime("%H:%M").isin(["09:35", "09:36", "09:37", "09:38"]).all()
    assert raw.timestamp.dt.second.eq(0).all() and raw.timestamp.dt.microsecond.eq(0).all()
    assert raw.timestamp.dt.strftime("%Y-%m-%d").eq(raw.date).all()
    assert raw.date.between("2024-01-01", "2025-12-31").all()
    assert not raw.duplicated(["date", "code", "timestamp"]).any()
    keys = pd.read_parquet(ROOT / "window_keys.parquet")
    assert len(raw.merge(keys, on=["date", "code"], validate="many_to_one")) == len(raw)
    groups = {key: p for key, p in raw.groupby(["date", "code"])}
    expected = []
    for day, code in keys.itertuples(index=False, name=None):
        p = groups.get((day, code))
        names = ["bars", "labels", "valid_bars", "positive_bars", "invalid_cent_bars",
                 "volume", "vwap", "positive_low", "positive_high"]
        r = dict(date=day, code=code, **dict.fromkeys(names, np.nan),
                 source_valid=False, queue_bounds_valid=False)
        if p is not None:
            valid = 0
            bad_cent = 0
            for b in p.itertuples(index=False):
                finite = all(math.isfinite(v) for v in [b.open, b.high, b.low, b.close, b.volume, b.amount])
                ohlc = (min(b.open, b.high, b.low, b.close) > 0
                        and b.high >= max(b.open, b.close, b.low) - .0001
                        and b.low <= min(b.open, b.close) + .0001)
                quantities = (b.volume >= 0 and b.amount >= 0 and ((b.volume == 0) == (b.amount == 0)))
                average = b.volume == 0 or b.low - .0101 <= b.amount / b.volume <= b.high + .0101
                valid += int(finite and ohlc and quantities and average)
                cent_ok = abs(b.high - round(b.high, 2)) <= .0001 and abs(b.low - round(b.low, 2)) <= .0001
                bad_cent += int(b.volume > 0 and not cent_ok)
            positive = p.loc[p.volume.gt(0)]
            volume = math.fsum(p.volume)
            r.update(bars=len(p), labels=p.timestamp.dt.strftime("%H:%M").nunique(),
                     valid_bars=valid, positive_bars=len(positive), invalid_cent_bars=bad_cent,
                     volume=volume, vwap=math.fsum(p.amount) / volume if volume else np.nan,
                     positive_low=positive.low.min(), positive_high=positive.high.max(),
                     source_valid=len(p) == 4 and p.timestamp.nunique() == 4 and valid == 4,
                     queue_bounds_valid=len(positive) > 0 and bad_cent == 0)
        expected.append(r)
    want = pd.DataFrame(expected).sort_values(["date", "code"]).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / "windows.parquet").sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[want.columns], want, check_dtype=False, atol=2e-12, rtol=0)

    # An independent Parquet reader re-opens a preselected 24 windows per half.
    sample = pd.read_parquet(ROOT / "old_tail_labels.parquet", columns=["date", "code", "half", "next_date"])
    sample["hash"] = [hashlib.sha256(("morning-source|" + d + "|" + c).encode()).hexdigest()
                      for d, c in zip(sample.date, sample.code)]
    sample = sample.sort_values(["half", "hash"]).groupby("half", sort=True).head(24)
    source_rows = 0
    columns = ["timestamp", "open", "high", "low", "close", "volume", "amount"]
    for r in sample.itertuples(index=False):
        file = MINUTES / r.code[:2].upper() / (r.code[3:] + ".parquet")
        start = pd.Timestamp(r.next_date + " 09:35:00")
        end = pd.Timestamp(r.next_date + " 09:39:00")
        p = pd.read_parquet(file, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
                            filters=[("timestamp", ">=", start), ("timestamp", "<", end)])
        p = p.rename(columns={"turnover": "amount"}).sort_values("timestamp").reset_index(drop=True)
        cache = raw.loc[raw.date.eq(r.next_date) & raw.code.eq(r.code), columns].sort_values("timestamp").reset_index(drop=True)
        pd.testing.assert_frame_equal(p[columns], cache, check_dtype=False, check_exact=True)
        source_rows += len(p)
    result = dict(passed=True, raw_report_sha256=sha(ROOT / "raw_report.json"),
                  raw_rows=len(raw), windows=len(want), independent_original_windows=len(sample),
                  independent_original_rows=source_rows, new_2026_prices_read=False)
    save_json(ROOT / "window_verification.json", result)
    return result


def dec(value):
    return Decimal(str(float(value)))


def cash_price(vwap, bps, side):
    if not math.isfinite(vwap):
        return None
    p = dec(vwap)
    spread = max(p * Decimal(bps) / Decimal(10000), Decimal(".005"))
    return p + spread if side == "buy" else p - spread


def account(r):
    """Scalar state machine independent of the vectorized production function."""
    out = {}
    ref_ok = math.isfinite(r.next_preclose) and r.next_preclose > 0 and abs(r.next_preclose - round(r.next_preclose, 2)) <= .0001
    close_ok = math.isfinite(r.day_close) and r.day_close > 0 and abs(r.day_close - round(r.day_close, 2)) <= .0001
    rate = 20 if r.board in ["chinext", "star"] else 5 if r.next_isST == 1 else 10
    lower = float((dec(r.next_preclose) * Decimal(100-rate) / Decimal(100)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)) if ref_ok else 0.
    daily_ok = ref_ok and r.next_trade_status == 1 and r.next_isST in [0, 1] and r.next_adjustflag == 3
    ref_gap = not (ref_ok and close_ok) or abs(r.next_preclose - r.day_close) > .005
    corporate = not r.catalog_covered or r.action_exposure or ref_gap
    entry_liquid = math.isfinite(r.entry_vwap) and r.entry_vwap > 0 and r.entry_volume > 0
    if not r.necessary_tradeable:
        entry = "not_submitted"
    elif not entry_liquid:
        entry = "no_liquidity"
    elif r.entry_volume < r.decision_shares * 10:
        entry = "volume_cap"
    elif dec(r.entry_vwap) * Decimal("1.0005") >= dec(r.upper_limit) - Decimal(".005"):
        entry = "estimated_upper_limit"
    else:
        entry = "filled"
    bought = entry == "filled"
    sealed = r.entry_bounds_valid and round(r.entry_low, 2) == r.upper_limit and round(r.entry_high, 2) == r.upper_limit
    entry_queue = bought and (not r.entry_bounds_valid or round(r.entry_high, 2) >= r.upper_limit)
    exit_liquid = math.isfinite(r.exit_vwap) and r.exit_vwap > 0 and r.exit_volume > 0
    if not bought:
        exit_state = "no_recorded_entry"
    elif not daily_ok:
        exit_state = "no_trading_bar"
    elif not exit_liquid:
        exit_state = "no_liquidity"
    elif r.exit_volume < r.decision_shares * 10:
        exit_state = "volume_cap"
    elif dec(r.exit_vwap) * Decimal(".9995") <= dec(lower) + Decimal(".005"):
        exit_state = "estimated_lower_limit"
    else:
        exit_state = "filled"
    exit_queue = bought and (not r.exit_bounds_valid or round(r.exit_low, 2) <= lower)
    no_entry = r.necessary_tradeable and r.entry_source_valid and not bought
    priorities = [(not r.necessary_tradeable, "not_submitted"), (not r.entry_source_valid, "entry_source_unknown"),
                  (no_entry, "not_bought"), (entry_queue, "entry_queue_unknown"), (corporate, "corporate_action_unknown"),
                  (not r.exit_source_valid, "exit_source_unknown"), (exit_state != "filled", "unknown_exit"),
                  (exit_queue, "exit_queue_unknown")]
    status = next((value for predicate, value in priorities if predicate), "ordinary_t1")
    source_bad = r.necessary_tradeable and (r.period_entry_bad_day or r.period_bad_symbol
                    or ((r.period_exit_bad_day or r.period_bad_symbol) and not no_entry))
    out.update(exit_lower_limit=lower, exit_daily_valid=daily_ok, reference_gap_or_unknown=ref_gap,
               corporate_unknown=corporate, entry_fill_status=entry, entry_recorded=bought,
               entry_sealed_limit=sealed, entry_queue_unknown=entry_queue, exit_fill_status=exit_state,
               exit_queue_unknown=exit_queue, original_base_status=status, period_source_unknown=source_bad,
               base_status="period_source_unknown" if source_bad and status in ["ordinary_t1", "not_bought"] else status)
    for bps in [5, 15]:
        buy = cash_price(r.entry_vwap, bps, "buy")
        sell = cash_price(r.exit_vwap, bps, "sell")
        stress = (buy is not None and buy >= dec(r.upper_limit) - Decimal(".005")) or (sell is not None and sell <= dec(lower) + Decimal(".005"))
        eligible = status == "ordinary_t1" and not stress
        buy_value = buy * int(r.decision_shares) if buy is not None else None
        planned = buy_value + max(Decimal(5), buy_value * Decimal(".0003")) + buy_value * Decimal(".00001") if buy is not None else None
        if eligible:
            sell_value = sell * int(r.decision_shares)
            net_cash = sell_value - max(Decimal(5), sell_value * Decimal(".0003")) - sell_value * Decimal(".00051")
            net = float(net_cash / planned - 1)
        else:
            net = np.nan
        label = ("not_submitted" if not r.necessary_tradeable else "no_trade" if no_entry else "unknown" if not eligible
                 else "economic_winner" if net >= .01 else "economic_loser" if net <= -.01 else "middle")
        out[f"stress_limit_unknown{bps}"] = stress
        out[f"original_label{bps}"] = label
        out[f"known_profit{bps}"] = eligible and not source_bad
        out[f"label{bps}"] = "unknown" if source_bad else label
        out[f"intended_buy_cash{bps}"] = float(planned) if planned is not None else np.nan
        out[f"buy_cash{bps}"] = float(planned) if eligible and not source_bad else np.nan
        out[f"sell_cash{bps}"] = float(net_cash) if eligible and not source_bad else np.nan
        out[f"net_return{bps}"] = net if eligible and not source_bad else np.nan
    return out


def labels(ROOT=ROOT):
    report = json.loads((ROOT / "label_report.json").read_text())
    assert report["labels_sha256"] == sha(ROOT / "labels.parquet")
    morning = pd.read_parquet(ROOT / "labels.parquet")
    tail = pd.read_parquet(ROOT / "old_tail_labels.parquet")
    win = pd.read_parquet(ROOT / "windows.parquet").set_index(["date", "code"])
    pd.testing.assert_frame_equal(tail[["date", "code"]], morning[["date", "code"]])
    mapping = {"bars": "exit_bars", "labels": "exit_labels", "source_valid": "exit_source_valid",
               "queue_bounds_valid": "exit_bounds_valid", "volume": "exit_volume", "vwap": "exit_vwap",
               "positive_low": "exit_low", "positive_high": "exit_high"}
    for r in morning.itertuples(index=False):
        w = win.loc[(r.next_date, r.code)]
        for key, col in mapping.items():
            value = getattr(r, col)
            assert (pd.isna(value) and pd.isna(w[key])) or value == w[key], (r.date, r.code, col)
    checked = 0
    calculated = set()
    for kind, frame in [("tail", tail), ("morning", morning)]:
        expected = pd.DataFrame([account(r) for r in frame.itertuples(index=False)])
        calculated.update(expected.columns)
        for col in expected:
            if col not in frame:
                assert kind == "tail" and col.startswith("intended_buy_cash")
                continue
            if pd.api.types.is_numeric_dtype(expected[col]) and not pd.api.types.is_bool_dtype(expected[col]):
                tolerance = 2e-8 if "cash" in col else 2e-12
                np.testing.assert_allclose(frame[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float),
                                           rtol=0, atol=tolerance, equal_nan=True, err_msg=kind + ":" + col)
            else:
                assert frame[col].eq(expected[col]).all(), (kind, col, frame.loc[frame[col].ne(expected[col]), ["date", "code", col]])
            checked += len(frame)
    unchanged = [col for col in tail if col not in calculated and col not in mapping.values()]
    pd.testing.assert_frame_equal(tail[unchanged], morning[unchanged], check_dtype=False, check_exact=True)
    result = dict(passed=True, label_report_sha256=sha(ROOT / "label_report.json"),
                  window_verification_sha256=sha(ROOT / "window_verification.json"),
                  tail_rows=len(tail), morning_rows=len(morning), scalar_fields_checked=checked,
                  unchanged_columns=len(unchanged), new_2026_prices_read=False)
    save_json(ROOT / "label_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["windows", "labels"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](ROOT=args.root), ensure_ascii=False, indent=2))
