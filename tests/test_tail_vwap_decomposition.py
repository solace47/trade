from pathlib import Path

import pandas as pd
import pytest

from trade_research.tail_vwap_decomposition import _inputs, _net_with_slippage


def test_input_decomposition_uses_only_predecision_bars(tmp_path: Path) -> None:
    candidate = pd.DataFrame({
        "date": ["2024-01-02"], "code": ["sh.600000"],
        "half": ["2024H1"], "board": ["main"],
        "price_1449": [10.1],
    })
    source = pd.DataFrame({
        "date": candidate.date, "code": candidate.code,
        "price_1435": [10.0], "vwap_1446_1449": [10.05],
        "price_1450": [100.0],
    })
    candidate_path = tmp_path / "candidates.parquet"
    anchor_dir = tmp_path / "prefix" / "2024"
    anchor_dir.mkdir(parents=True)
    candidate.to_parquet(candidate_path)
    source.to_parquet(anchor_dir / "part_000.parquet")
    first = _inputs(candidate_path, anchor_dir.parent)
    source.price_1450 = 1.0
    source.to_parquet(anchor_dir / "part_000.parquet")
    second = _inputs(candidate_path, anchor_dir.parent)
    pd.testing.assert_frame_equal(first, second)
    assert first.robust_pp.iloc[0] == pytest.approx(.5)
    assert first.noise_pp.iloc[0] == pytest.approx(100 * (10.1 / 10.05 - 1))


def test_cost_repricing_matches_five_basis_point_archive_formula() -> None:
    raw_entry, raw_exit, shares = 10.0, 10.2, 10_000
    frame = pd.DataFrame({
        "entry_price": [raw_entry * 1.0005],
        "exit_price": [raw_exit * .9995],
        "shares": [shares],
    })
    buy = shares * raw_entry * 1.0005
    sell = shares * raw_exit * .9995
    expected = ((sell - max(5, sell * .0003) - sell * .00001 - sell * .0005)
                / (buy + max(5, buy * .0003) + buy * .00001) - 1)
    assert _net_with_slippage(frame, 5)[0] == pytest.approx(expected)
    assert _net_with_slippage(frame, 15)[0] < expected
