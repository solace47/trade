import numpy as np
import pandas as pd
import pytest

from trade_research.turnover_reference import attach_references, history_states, summarize


def history():
    return pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "code": ["sh.600001"] * 3, "close": [10., 12., 6.6], "preclose": [10., 10., 6.],
        "turn": [10., 20., 25.], "volume": [1000, 2000, 2500],
        "tradestatus": [1] * 3, "adjustflag": [3] * 3})


def signal():
    return pd.DataFrame({"date": ["2024-01-05"], "code": ["sh.600001"], "half": ["2024H1"],
        "previous_traded_date": ["2024-01-04"], "prior_sessions": [3],
        "previous_close": [6.6], "preclose": [6.6], "price_1449": [7.]})


def test_explicit_weights_and_price_scale():
    h = history()
    s = history_states(h, h.date.tolist()).iloc[-1]
    assert s.initial_mass == pytest.approx(.9 * .8 * .75)
    # Earlier historical prices are halved on the third day.
    assert s.known_contribution == pytest.approx(.1 * .8 * .75 * 5 + .2 * .75 * 6 + .25 * 6.6)
    assert s.initial_contribution == pytest.approx(5 * .9 * .8 * .75)
    assert s.price_reference_changes == 1


def test_future_day_does_not_change_prior_signal():
    h = history()
    s = history_states(h, h.date.tolist())
    expected = attach_references(signal(), s)
    future = h.iloc[[-1]].assign(date="2024-01-05", close=100., turn=100., preclose=6.6)
    extended = pd.concat([h, future], ignore_index=True)
    got = attach_references(signal(), history_states(extended, extended.date.tolist()))
    pd.testing.assert_frame_equal(got, expected)
    same_day = signal().assign(previous_traded_date="2024-01-05", prior_sessions=4, previous_close=100.)
    with pytest.raises(ValueError, match="contemporaneous"):
        attach_references(same_day, history_states(extended, extended.date.tolist()))


def test_full_turnover_removes_initialization():
    h = history()
    h.loc[1, "turn"] = 100.
    out = attach_references(signal(), history_states(h, h.date.tolist())).iloc[0]
    assert out.initial_mass == 0
    assert out.initial_contribution_asof == 0
    assert out.gain_scenario_width == 0
    assert not out.gain_sign_varies


def test_unknown_prior_transition_is_not_filled_or_clipped():
    for bad in (np.nan, -1., 101., 0.):
        h = history()
        h.loc[0, "turn"] = bad
        s = history_states(h, h.date.tolist())
        out = attach_references(signal(), s)
        assert not out.reference_valid.any()
        assert out.gain_normalized.isna().all()
        assert summarize(out)[0]["unknown_rows"] == 1


def test_suspension_does_not_update_turnover_but_missing_date_is_unknown():
    h = history()
    h.loc[1, ["tradestatus", "volume", "turn", "close"]] = [0, 0, np.nan, 10.]
    s = history_states(h, h.date.tolist())
    assert s.iloc[-1].history_valid
    assert s.iloc[-1].initial_mass == pytest.approx(.9 * .75)
    s = history_states(h.drop(index=1), h.date.tolist())
    assert not s.iloc[-1].history_valid
    assert s.iloc[-1].missing_history_sessions == 1


def test_range_identity_and_denominator_preserved():
    h = history()
    base = pd.concat([signal(), signal().assign(code="sz.000001")], ignore_index=True)
    out = attach_references(base, history_states(h, h.date.tolist()))
    good = out.loc[out.reference_valid].iloc[0]
    assert good.gain_scenario_width == pytest.approx(1.5 * good.initial_contribution_asof / good.price_1449)
    assert len(out) == 2
    row = summarize(out)[0]
    assert row["all_rows"] == 2 and row["unknown_rows"] == 1


def test_history_year_and_duplicate_guard():
    h = history()
    with pytest.raises(ValueError, match="permitted years"):
        history_states(h.assign(date="2026-01-02").iloc[:1], ["2026-01-02"])
    with pytest.raises(ValueError, match="Duplicate"):
        history_states(pd.concat([h, h]), h.date.tolist())


def test_verified_empty_suspension_volume_is_not_a_missing_trade():
    h = history()
    h["volume"] = h.volume.astype("Int64")
    h.loc[1, ["tradestatus", "volume", "turn", "close"]] = [0, pd.NA, np.nan, 10.]
    s = history_states(h, h.date.tolist())
    assert s.iloc[-1].history_valid
    assert s.iloc[-1].invalid_history_rows == 0
    assert s.iloc[-1].initial_mass == pytest.approx(.9 * .75)
    h.loc[1, "volume"] = 100
    assert not history_states(h, h.date.tolist()).iloc[-1].history_valid
