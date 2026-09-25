from scripts.propose_pledge_table_crossings import _table_rows


def test_table_row_uses_printed_ratio_to_confirm_crossing() -> None:
    table = [
        ["股东名称", "持股数量（股）", "本次解质押前质押股数",
         "本次解质押后质押股数", "占其所持股份比例"],
        ["直接控股股东", "36,680,617", "21,930,000", "17,930,000",
         "48.88%"],
    ]
    result = _table_rows(table)
    assert len(result) == 1
    assert result[0]["before_pledged"] == 21_930_000
    assert result[0]["crossed_tier"] == "50"
    table[1][4] = "54.19%"
    assert _table_rows(table) == []


def test_table_row_scales_ten_thousand_share_units() -> None:
    table = [
        ["股东名称", "持股数量（万股）", "质押前数量（万股）",
         "质押后数量（万股）", "占其所持股份比例"],
        ["直接控股股东", "18,151.9262", "10,100", "9,000", "49.58%"],
    ]
    result = _table_rows(table)
    assert len(result) == 1
    assert result[0]["held"] == 181_519_262
    assert result[0]["after_pledged"] == 90_000_000


def test_unchanged_pledge_is_not_mistaken_for_later_percent_columns() -> None:
    table = [
        ["股东名称", "持股数量", "持股比例", "解质押前质押", "解质押后质押",
         "占其所持股份比例", "占公司总股本", "冻结股份"],
        ["李琳", "3,689,762", "2.60", "3,689,762", "3,689,762",
         "100.0000", "2.5969", "0"],
    ]
    assert _table_rows(table) == []


def test_multiple_actor_rows_are_retained_for_manual_review() -> None:
    table = [
        ["股东名称", "持股数量", "质押前数量", "质押后数量",
         "占其所持股份比例"],
        ["甲", "10,000,000", "6,000,000", "4,000,000", "40%"],
        ["乙", "10,000,000", "9,000,000", "7,000,000", "70%"],
    ]
    assert [row["actor_cell"] for row in _table_rows(table)] == ["甲", "乙"]
