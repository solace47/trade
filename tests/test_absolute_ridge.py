"""Protect cash abstention, cutoff timing and same-board control matching."""

import duckdb
import pandas as pd
import pytest

from trade_research.absolute_ridge import choose, match_controls, training_labels


def test_choose_holds_cash_and_respects_five_session_cooldown() -> None:
    scores = pd.DataFrame([
        {"date": "2024-07-01", "code": "sh.600001", "score": .02},
        {"date": "2024-07-01", "code": "sh.600002", "score": -.01},
        {"date": "2024-07-02", "code": "sh.600001", "score": .03},
        {"date": "2024-07-03", "code": "sh.600003", "score": -.01},
    ])

    result = choose(scores, ["2024-07-01", "2024-07-02", "2024-07-03"])

    assert result[["date", "code"]].to_records(index=False).tolist() == [
        ("2024-07-01", "sh.600001")]


def test_match_controls_stays_on_board_and_rejects_return_gap() -> None:
    date = "2024-07-01"
    chosen = pd.DataFrame([
        {"date": date, "code": "sh.600001", "daily_rank": 1,
         "score": .01},
    ])
    rows = (
        ("sh.600001", "main", .01),
        ("sz.300001", "chinext", .01),
        ("sh.600002", "main", .015),
        ("sh.600003", "main", .05),
    )
    features = pd.DataFrame([
        {"date": date, "code": code, "board": board,
         "amount_1450": 200_000_000, "price_1450": 10.0,
         "return20_prior_adjusted": 0.0, "return_1450": day_return}
        for code, board, day_return in rows
    ])

    controls = match_controls(chosen, features)

    assert controls.code.tolist() == ["sh.600002"]
    assert controls.pair_id.tolist() == ["sh.600001"]


def test_training_rejects_an_exit_inside_the_test_period() -> None:
    connection = duckdb.connect()
    connection.register("outcomes", pd.DataFrame([
        {"date": "2024-06-14", "code": "sh.600001", "horizon": 5,
         "exit_date": "2024-07-02", "net_return": .01,
         "exit_status": "filled", "exit_delay_sessions": 0},
    ]))
    connection.register("bad_days", pd.DataFrame({
        "date": pd.Series(dtype=str), "code": pd.Series(dtype=str),
    }))
    train = pd.DataFrame([
        {"date": "2024-06-14", "code": "sh.600001"},
    ])

    with pytest.raises(ValueError, match="overlap test time"):
        training_labels(connection, train, "2024-07-01")
