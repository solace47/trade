from scripts.propose_pledge_crossings import vertical_crossing


def test_original_counts_cross_eighty_with_five_point_release() -> None:
    text = ("本次解质股份（股） 6,500,000\n"
            "持股数量（股） 41,607,455\n"
            "剩余被质押股份数量（股） 27,000,000")
    result = vertical_crossing(text)
    assert result is not None
    assert result["crossed_tier"] == "80"
    assert result["released"] == 6_500_000
    assert result["after_pct"] == 64.8922


def test_no_crossing_or_inconsistent_counts_are_not_proposals() -> None:
    template = ("本次解质股份（股） {released}\n"
                "持股数量（股） 100\n"
                "剩余被质押股份数量（股） {remaining}")
    assert vertical_crossing(template.format(released=4, remaining=48)) is None
    assert vertical_crossing(template.format(released=10, remaining=55)) is None
    assert vertical_crossing(template.format(released=60, remaining=50)) is None


def test_ten_thousand_share_units_and_fractional_counts() -> None:
    table_unit = ("单位：万股 股东名称 胥爱民 "
                  "本次解质股份 610 持股数量 3,690.5021 "
                  "剩余被质押股份数量 1,556")
    result = vertical_crossing(table_unit)
    assert result is not None
    assert result["held"] == 36_905_021
    assert result["released"] == 6_100_000
    explicit_units = ("本次解质股份 5,000万股 "
                      "持股数量 20,005.1612万股 "
                      "剩余被质押股份数量 9,350万股")
    result = vertical_crossing(explicit_units)
    assert result is not None
    assert result["held"] == 200_051_612
    assert result["remaining"] == 93_500_000
