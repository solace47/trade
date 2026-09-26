"""Freeze same-day loser comparisons conditional on stable reference gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .cash_ex_matching import assess, match_candidates
from .corporate_cash import save_json, sha
from .quote_precision import quote_cents
from .turnover_reference import ROOT as SOURCE


ROOT = Path("data/research/reference_gain_inputs")
BARS = Path("data/research/noon_recovery_1449/bars")
INPUT_SHA = "1bd59389f98301ee05cf86ab3b2478375e726f0f57755d688a583a8ff30ebde5"
RULE_COMMIT = "082470f"


def classify(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    quotes = []
    for value in out.price_1449:
        try:
            quotes.append(quote_cents(value) / 100)
        except ValueError:
            quotes.append(np.nan)
    # Price ratios use the cent-restored decision quote. Reference costs and
    # the earlier 14:20 anchor retain their original, already-known values.
    anchor_1420 = out.price_1449 / (1 + out.return_last29)
    out["original_price_1449"] = out.price_1449
    out["price_1449"] = quotes
    out["day_return"] = out.price_1449 / out.preclose - 1
    out["return_last29"] = out.price_1449 / anchor_1420 - 1
    out["screen_gain_low"] = 1 - out.reference_seed_double / out.price_1449
    out["screen_gain_high"] = 1 - out.reference_seed_half / out.price_1449
    out["screen_width"] = out.screen_gain_high - out.screen_gain_low
    out["raw_quote_valid"] = (out.n_1449.eq(1) & out.valid_1449.fillna(False)
        & np.isfinite(out.price_1449)
        & (out.raw_price_1449 - out.price_1449).abs().le(.0001))
    out["screen_eligible"] = (out.reference_valid & out.screen_width.le(.01)
        & out.day_return.between(-.03, -.005) & ~out.reference_gap & out.raw_quote_valid
        & out.prior20_turnover_count.eq(20) & out.prior20_turnover_pct.gt(0)
        & np.isfinite(out.prior20_turnover_pct))
    out["reference_group"] = "outside"
    out.loc[out.screen_eligible & out.screen_gain_low.ge(.02), "reference_group"] = "gain"
    out.loc[out.screen_eligible & out.screen_gain_high.le(-.02), "reference_group"] = "loss"
    candidate = out.loc[out.reference_group.eq("gain")].sort_values(
        ["date", "screen_gain_low", "code"], ascending=[True, False, True]).copy()
    candidate["daily_rank"] = candidate.groupby("date", sort=False).cumcount() + 1
    controls = out.loc[out.reference_group.eq("loss")].copy()
    return out, candidate, controls


def assess_pairs(attempts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    result = assess(attempts, pairs)
    for row in result:
        part = pairs.loc[pairs.half.eq(row["half"])]
        ratio = float(part.prior20_turnover_pct_ratio.mean()) if len(part) else None
        row["prior20_turnover_mean_ratio"] = ratio
        row["checks"].update({"pairs": len(part) >= 120, "pair_days": part.date.nunique() >= 40,
            "prior20_turnover_balance": ratio is not None and .8 <= ratio <= 1.25,
            "group_and_width_valid": bool(len(part)
                and part.screen_gain_low.ge(.02).all()
                and part.control_screen_gain_high.le(-.02).all()
                and part.screen_width.le(.01).all()
                and part.control_screen_width.le(.01).all())})
        for feature in ("screen_gain_low", "screen_gain_high", "screen_width", "prior20_turnover_pct"):
            for prefix in ("", "control_"):
                row[prefix + feature + "_mean"] = float(part[prefix + feature].mean()) if len(part) else None
        row["passed"] = all(row["checks"].values())
    return result


def build(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((SOURCE / "report.json").read_text())
    hashes = {str(SOURCE / "input_references.parquet"): INPUT_SHA,
              str(SOURCE / "history_inputs.parquet"): previous["output_sha256"]["history_inputs.parquet"]}
    hashes.update({str(p): sha(p) for p in sorted(BARS.glob("part_*.parquet"))})
    if len(hashes) <= 2:
        raise ValueError("Raw decision bar cache missing")
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen input changed")
    manifest = {"rule_commit": RULE_COMMIT, "inputs_sha256": hashes,
                "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Cannot replace frozen reference-gain inputs")
    save_json(output / "manifest.json", manifest)
    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        con.read_parquet(str(SOURCE / "input_references.parquet")).create_view("signals")
        con.read_parquet(str(SOURCE / "history_inputs.parquet")).create_view("history")
        con.read_parquet([str(p) for p in sorted(BARS.glob("part_*.parquet"))]).create_view("bars")
        con.execute("""
            CREATE TEMP VIEW turnover AS
            SELECT date AS source_date, code, COUNT(turn) OVER w AS prior20_turnover_count,
                   AVG(turn) OVER w AS prior20_turnover_pct
            FROM history WHERE tradestatus=1
            WINDOW w AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        """)
        frame = con.execute("""
            SELECT s.*, t.prior20_turnover_count, t.prior20_turnover_pct,
                   b.n_1449, b.raw_price_1449, b.valid_1449
            FROM signals s LEFT JOIN turnover t USING (code, source_date)
            LEFT JOIN bars b USING (code, date)
            WHERE s.date BETWEEN '2024-01-01' AND '2025-12-31'
            ORDER BY s.date,s.code
        """).df()
    finally:
        con.close()
    if len(frame) != 328895 or frame.duplicated(["date", "code"]).any():
        raise ValueError("Input association changed the frozen denominator")
    if not frame.source_date.lt(frame.date).all():
        raise ValueError("Turnover is not strictly lagged")
    screened, candidates, controls = classify(frame)
    attempts = candidates.loc[candidates.daily_rank.le(5)].copy()
    pairs, missed = match_candidates(attempts, controls,
        extra_ratio_limits={"prior20_turnover_pct": 2.0})
    diagnostics = ["screen_gain_low", "screen_gain_high", "screen_width"]
    pairs = pairs.merge(attempts[["date", "code", *diagnostics]], on=["date", "code"], validate="one_to_one")
    pairs = pairs.merge(controls[["date", "code", *diagnostics]].rename(columns={
        "code": "control_code", **{name: "control_" + name for name in diagnostics}}),
        on=["date", "control_code"], validate="one_to_one")
    if (len(pairs) + len(missed) != len(attempts) or pairs.duplicated(["date", "control_code"]).any()):
        raise ValueError("Lost attempts or reused controls")
    association = []
    for half, group in screened.groupby("half", sort=True):
        present = group.n_1449.eq(1)
        aligned = (group.raw_price_1449 - group.original_price_1449).abs().le(.0001)
        covered = present & aligned & group.prior20_turnover_count.eq(20)
        association.append({"half": half, "all_rows": len(group),
            "associated_rows": int(covered.sum()), "association_fraction": float(covered.mean()),
            "raw_quote_valid_rows": int(group.raw_quote_valid.sum()),
            "screen_eligible_rows": int(group.screen_eligible.sum()),
            "gain_rows": int(group.reference_group.eq("gain").sum()),
            "loss_rows": int(group.reference_group.eq("loss").sum()), "passed": bool(covered.mean() >= .999)})
    summaries = assess_pairs(attempts, pairs)
    tables = {"screened_inputs": screened, "all_candidates": candidates, "controls": controls,
              "top_signals": attempts, "input_pairs": pairs, "unmatched_attempts": missed}
    for name, table in tables.items():
        table.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    result = {"rule_commit": RULE_COMMIT, "association": association, "by_half": summaries,
              "input_gate_passed": len(association) == 4 and all(r["passed"] for r in association + summaries),
              "output_sha256": {name: sha(output / f"{name}.parquet") for name in tables},
              "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    save_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    print(build(parser.parse_args().output))


if __name__ == "__main__":
    main()
