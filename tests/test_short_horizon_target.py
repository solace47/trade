import pandas as pd
import pytest

from trade_research.short_horizon_labels import align_risk_scores
from trade_research.short_horizon_target import training_fold


def test_unknown_queue_cannot_provide_a_positive_training_target():
    rows = pd.DataFrame({"date": ["2024-01-02"]*3, "exit_date": ["2024-01-03"]*3,
        "score_origin": ["recorded_economic_scenario", "known_not_bought", "unresolved_full_loss_training_scenario"],
        "entry_price": [10.005, None, 10.005], "exit_price": [10.9945, None, None],
        "shares": [2000, 0, 2000], "sold_shares": [2000, 0, 2000],
        "dividend_gross": [0., 0., 0.], "dividend_tax": [0., 0., 0.],
        "downside_score": [.09, 0., -1.], "economic_scenario15": [.09, None, None]})
    checked = align_risk_scores(rows, pd.Series([True, False, True]))
    assert checked.downside_score.tolist() == [-1., 0., -1.]
    assert checked.loc[0, "economic_scenario15"] > 0  # Conditional value kept separately.
    assert checked.loc[0, "score_origin"] == "adverse_limit_queue_training_penalty"
    assert checked.loc[2, "score_origin"] == "unresolved_full_loss_training_scenario"


def test_a_training_dividend_cannot_cross_the_fit_boundary():
    calendar = pd.bdate_range("2023-12-01", periods=30).strftime("%Y-%m-%d").tolist()
    features = pd.DataFrame([{"date": calendar[0], "code": "sh.600000"}])
    labels = features.assign(downside_score=.01, score_origin="recorded_economic_scenario",
        target_exit_date=calendar[1], exit_date=calendar[1], action_status="catalogue_scenario")
    fold = {"train_first": calendar[0], "train_last": calendar[0], "test_first": calendar[11]}
    catalog = pd.DataFrame([{"code": "sh.600000", "dividOperateDate": calendar[1], "dividPayDate": calendar[11]}])
    with pytest.raises(ValueError, match="dividend payment"):
        training_fold(features, labels, fold, calendar, catalog)
    catalog["dividPayDate"] = calendar[2]
    _, audit = training_fold(features, labels, fold, calendar, catalog)
    assert audit["last_used_dividend_pay_date"] == calendar[2]
    fold["test_first"] = calendar[10]
    with pytest.raises(ValueError, match="overlaps"):
        training_fold(features, labels, fold, calendar, catalog)
