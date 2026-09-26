"""Qlib's frozen Alpha158 expressions on an as-of 14:49 temporary daily bar.

Expression definitions and operator semantics derive from Microsoft Qlib;
see config/alpha158_definition.json and licenses/qlib-MIT.txt. This adapter
does not import Qlib, use its labels, or replace the project's execution model.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

DEFINITION = Path("config/alpha158_definition.json")
WINDOW = 61
FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@lru_cache(maxsize=1)
def definitions() -> tuple[tuple[str, ast.AST], ...]:
    data = json.loads(DEFINITION.read_text())
    result = tuple((r["name"], ast.parse(re.sub(r"\$(\w+)", r"\1", r["expression"]), mode="eval").body)
                   for r in data["fields"])
    if len(result) != 158 or len({name for name, _ in result}) != 158:
        raise ValueError("The complete fixed factor library is required")
    return result


class Evaluator:
    """Evaluate only explicitly supported arithmetic; no dynamic Python eval."""

    def __init__(self, banks: dict[str, np.ndarray], lengths: np.ndarray):
        self.banks, self.lengths, self.cache = banks, lengths, {}
        if set(banks) != set(FIELDS) or any(v.shape != (len(lengths), WINDOW) for v in banks.values()):
            raise ValueError("Expected current plus sixty historical bars")

    def value(self, node: ast.AST, offsets: tuple[int, ...] = (0,)) -> np.ndarray:
        key = ast.dump(node), offsets
        if key in self.cache:
            return self.cache[key]
        if not offsets or min(offsets) < 0 or max(offsets) >= WINDOW:
            raise ValueError("A factor requests future or unsupported historical information")
        if isinstance(node, ast.Name) and node.id in FIELDS:
            result = self.banks[node.id][:, offsets]
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            result = np.asarray(float(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            result = -self.value(node.operand, offsets)
        elif isinstance(node, ast.BinOp):
            operations = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
            if type(node.op) not in operations:
                raise ValueError("Unsupported arithmetic")
            result = operations[type(node.op)](self.value(node.left, offsets), self.value(node.right, offsets))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            operations = {ast.Gt: np.greater, ast.Lt: np.less}
            if type(node.ops[0]) not in operations:
                raise ValueError("Unsupported comparison")
            result = operations[type(node.ops[0])](self.value(node.left, offsets), self.value(node.comparators[0], offsets)).astype(float)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == "Ref":
                shift = ast.literal_eval(node.args[1])
                if not isinstance(shift, int) or shift < 0:
                    raise ValueError("Future reference is forbidden")
                result = self.value(node.args[0], tuple(offset + shift for offset in offsets))
            elif name in ("Abs", "Log", "Greater", "Less"):
                func = {"Abs": np.abs, "Log": np.log, "Greater": np.maximum, "Less": np.minimum}[name]
                result = func(*(self.value(arg, offsets) for arg in node.args))
            else:
                result = self.rolling(name, node.args, offsets)
        else:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        result = np.broadcast_to(result, (len(self.lengths), len(offsets))).copy()
        result[np.asarray(offsets)[None, :] >= self.lengths[:, None]] = np.nan
        self.cache[key] = result
        return result

    def rolling(self, name: str, args: list[ast.AST], offsets: tuple[int, ...]) -> np.ndarray:
        n = ast.literal_eval(args[2] if name == "Corr" else args[1])
        if n not in (5, 10, 20, 30, 60):
            raise ValueError("Rolling window is outside the frozen library")
        expanded = tuple(offset + lag for offset in offsets for lag in range(n - 1, -1, -1))
        shape = (len(self.lengths), len(offsets), n)
        x = self.value(args[0], expanded).reshape(shape)
        count = np.sum(~np.isnan(x), axis=-1)
        if name == "Mean":
            return np.nanmean(x, axis=-1)
        if name == "Sum":
            return np.where(count > 0, np.nansum(x, axis=-1), np.nan)
        if name == "Std":
            return np.nanstd(x, axis=-1, ddof=1)
        if name in ("Max", "Min"):
            return (np.nanmax if name == "Max" else np.nanmin)(x, axis=-1)
        if name == "Quantile":
            return np.nanquantile(x, ast.literal_eval(args[2]), axis=-1)
        if name == "Rank":
            last = x[..., -1:]
            rank = (np.sum(x < last, axis=-1) + (np.sum(x == last, axis=-1) + 1) / 2) / count
            return np.where(np.isnan(last[..., 0]), np.nan, rank)
        if name in ("IdxMax", "IdxMin"):
            live = (np.asarray(expanded)[None, :] < self.lengths[:, None]).reshape(shape)
            # Padding before listing is not a row in the source rolling series.
            values = np.where(live, x, -np.inf if name == "IdxMax" else np.inf)
            position = (np.argmax if name == "IdxMax" else np.argmin)(values, axis=-1)
            start = n - live.sum(axis=-1)
            return np.where(count > 0, position - start + 1, np.nan)
        if name in ("Slope", "Rsquare", "Resi"):
            valid = np.isfinite(x)
            t = np.broadcast_to(np.arange(1, n + 1, dtype=float), shape)
            tmean = np.sum(np.where(valid, t, 0), axis=-1) / valid.sum(axis=-1)
            ymean = np.nanmean(np.where(valid, x, np.nan), axis=-1)
            dx, dy = t - tmean[..., None], x - ymean[..., None]
            xx = np.sum(np.where(valid, dx * dx, 0), axis=-1)
            xy = np.sum(np.where(valid, dx * dy, 0), axis=-1)
            slope = xy / xx
            if name == "Slope":
                return slope
            if name == "Resi":
                return x[..., -1] - (ymean + slope * (n - tmean))
            yy = np.sum(np.where(valid, dy * dy, 0), axis=-1)
            rsquare = xy * xy / (xx * yy)
            return np.where(np.isclose(np.nanstd(x, axis=-1, ddof=1), 0, atol=2e-5), np.nan, rsquare)
        if name == "Corr":
            y = self.value(args[1], expanded).reshape(shape)
            valid = np.isfinite(x) & np.isfinite(y)
            a, b = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
            dx, dy = a - np.nanmean(a, axis=-1)[..., None], b - np.nanmean(b, axis=-1)[..., None]
            corr = np.nansum(dx * dy, axis=-1) / np.sqrt(np.nansum(dx * dx, axis=-1) * np.nansum(dy * dy, axis=-1))
            flat = np.isclose(np.nanstd(x, axis=-1, ddof=1), 0, atol=2e-5) | np.isclose(np.nanstd(y, axis=-1, ddof=1), 0, atol=2e-5)
            return np.where(flat | (valid.sum(axis=-1) < 2), np.nan, corr)
        raise ValueError(f"Unsupported rolling operator: {name}")

    def frame(self) -> pd.DataFrame:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = {name: self.value(node)[:, 0] for name, node in definitions()}
        return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan)


def asof_banks(daily: pd.DataFrame, prefix: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Every row gets completed prior days and its own current partial bar."""
    if daily.duplicated("date").any() or prefix.duplicated("date").any():
        raise ValueError("Duplicate per-security dates")
    d = daily.loc[pd.to_numeric(daily.tradestatus).eq(1)].sort_values("date").reset_index(drop=True).copy()
    for name in (*FIELDS[:-1], "preclose", "amount"):
        d[name] = pd.to_numeric(d[name], errors="coerce")
    previous = d.close.shift(1)
    ratio = (d.preclose / previous).where((d.preclose - previous).abs().gt(.005), 1.).fillna(1.)
    factor = ratio.cumprod().to_numpy(float)
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError("Invalid causal reference factor")
    indices = pd.Index(d.date).get_indexer(prefix.date)
    if (indices < 0).any():
        raise ValueError("A decision prefix has no actual traded daily row")
    lengths = indices + 1
    take = indices[:, None] - np.arange(WINDOW)[None, :]
    outside = take < 0
    take = np.maximum(take, 0)
    d["vwap"] = d.amount / d.volume.where(d.volume.gt(0))
    banks = {}
    for name in FIELDS:
        source = d[name].to_numpy(float) * factor if name == "volume" else d[name].to_numpy(float) / factor
        bank = source[take].copy()
        bank[outside] = np.nan
        if name == "open":
            now = d.open.to_numpy(float)[indices]
        elif name == "vwap":
            now = (prefix.amount_1449 / prefix.volume_1449).to_numpy(float)
        else:
            field = {"close": "price_1449", "high": "high_1449", "low": "low_1449", "volume": "volume_1449"}[name]
            now = prefix[field].to_numpy(float)
        bank[:, 0] = now * factor[indices] if name == "volume" else now / factor[indices]
        banks[name] = bank
    return banks, lengths


def features_for_symbol(daily: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    banks, lengths = asof_banks(daily, prefix)
    result = Evaluator(banks, lengths).frame()
    result.insert(0, "date", prefix.date.to_numpy())
    return result
