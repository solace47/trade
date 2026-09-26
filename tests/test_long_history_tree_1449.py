import pandas as pd
import pytest

from trade_research.long_history_tree_1449 import training_block


def inputs():
    calendar = pd.bdate_range("2024-06-03", periods=20).strftime("%Y-%m-%d").tolist()
    features = pd.DataFrame({"date": [calendar[0], calendar[1]], "code": ["sh.600000", "sh.600001"]})
    labels = features.assign(downside_score=[-.01, -1.],
        score_origin=["recorded_economic_scenario", "unresolved_full_loss_training_scenario"],
        target_exit_date=[calendar[5], calendar[6]], exit_date=[calendar[5], None])
    return features, labels, calendar


def test_unknown_training_scenario_is_not_dropped_or_replaced_with_zero():
    features, labels, calendar = inputs()
    train, endpoint = training_block(features, labels, calendar[1], calendar[12], calendar)
    assert train.downside_score.tolist() == [-.01, -1.]
    assert train.exit_date.isna().sum() == 1
    assert endpoint == calendar[11]


def test_early_recorded_exit_does_not_shorten_required_label_window():
    features, labels, calendar = inputs()
    labels["exit_date"] = calendar[6]
    with pytest.raises(ValueError, match="observation windows overlap"):
        training_block(features, labels, calendar[1], calendar[10], calendar)


def test_missing_stock_label_cannot_silently_shrink_training():
    features, labels, calendar = inputs()
    with pytest.raises(ValueError, match="Incomplete long-history"):
        training_block(features, labels.iloc[:1], calendar[1], calendar[12], calendar)


def test_actual_late_exit_is_checked_in_addition_to_planned_window():
    features, labels, calendar = inputs()
    labels.loc[0, "exit_date"] = calendar[13]
    with pytest.raises(ValueError, match="overlap the test"):
        training_block(features, labels, calendar[1], calendar[12], calendar)
