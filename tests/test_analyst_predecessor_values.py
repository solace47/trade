from decimal import Decimal

from trade_research.analyst_predecessor_values import compare_forecasts


def test_old_narrative_rounding_is_separate_from_continuous_forecast_change():
    value = compare_forecasts(Decimal("11143.4"), Decimal("11143"), "111.43")
    assert value["reported_previous_rounding_matches"]
    assert value["paired_forecast_direction"] == "down"
    assert Decimal(value["paired_forecast_relative_change"]) < 0
    assert compare_forecasts(Decimal("11143.5"), Decimal("11143"), "111.43")["reported_previous_rounding_matches"] is False


def test_missing_old_claim_or_nonpositive_denominator_is_not_filled():
    value = compare_forecasts(Decimal("100"), Decimal("110"), None)
    assert value["reported_previous_rounding_matches"] is None
    assert Decimal(value["paired_forecast_relative_change"]) == Decimal("0.1")
    for prior in ("0", "-100"):
        assert compare_forecasts(Decimal(prior), Decimal("110"), None)["paired_forecast_relative_change"] is None


def test_small_changes_with_different_printed_precision_are_flagged():
    for before, after in [("358", "357.94"), ("726", "726.27")]:
        result = compare_forecasts(Decimal(before), Decimal(after), None)
        assert result["display_precision_diagnostic"] == "overlapping_display_rounding_intervals"
    assert compare_forecasts(Decimal("326.79"), Decimal("332.19"), None)["display_precision_diagnostic"] == "up_beyond_display_rounding"
