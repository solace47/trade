from scripts.propose_pledge_after_only import after_only_crossings


def test_after_only_table_can_propose_direct_tier_crossing() -> None:
    release = [
        ["股东名称", "本次解除质押数量（万股）", "占其所持股份比例"],
        ["锦隆能源", "19,750.00", "21.63%"],
        ["合计", "20,925.00", "22.91%"],
    ]
    after = [
        ["股东名称", "持股数量（万股）", "目前累计质押数量（万股）",
         "占控股股东及一致行动人所持股份比例"],
        ["锦隆能源", "91,317.54", "25,619.10", "13.11%"],
    ]
    rows = after_only_crossings([[release, after]])
    assert len(rows) == 1
    assert rows[0]["released"] == 209_250_000
    assert rows[0]["held"] == 913_175_400
    assert rows[0]["crossed_tier"] == "50"


def test_after_only_table_rejects_weak_or_unchecked_ratio() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "4,000,000", "4.00%"],
    ]
    after = [
        ["股东名称", "持股数量（股）", "累计被质押数量（股）",
         "占其所持股份比例"],
        ["直接控股股东", "100,000,000", "49,000,000", "49%"],
    ]
    assert after_only_crossings([[release, after]]) == []
    release[1][2] = "40.00%"
    assert after_only_crossings([[release, after]]) == []


def test_after_only_table_sums_merged_actor_release_rows() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "22,000,000", "22.00%"],
        [None, "6,100,000", "6.10%"],
    ]
    after = [
        ["股东名称", "持股数量（股）", "累计被质押数量（股）",
         "占其所持股份比例"],
        ["直接控股股东", "100,000,000", "30,000,000", "30%"],
    ]
    rows = after_only_crossings([[release, after]])
    assert {row["released"] for row in rows} == {22_000_000, 28_100_000}
    assert all(row["crossed_tier"] == "50" for row in rows)


def test_extension_schedule_is_not_a_release() -> None:
    extension = [
        ["股东名称", "本次质押数量（万股）", "占其所持股份比例", "质押到期日"],
        ["直接控股股东", "1,871", "28.0078%", "至办理解除质押手续为止"],
    ]
    after = [
        ["股东名称", "持股数量（万股）", "累计被质押数量（万股）",
         "占其所持股份比例"],
        ["直接控股股东", "6,680.2874", "1,871", "28.0078%"],
    ]
    assert after_only_crossings([[extension, after]]) == []


def test_multi_actor_total_does_not_hide_controllers_own_sum() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "5,800,000", "12.93%"],
        [None, "5,500,000", "12.26%"],
        [None, "5,500,000", "12.26%"],
        ["实际控制人", "5,000,000", "49.96%"],
        ["合计", "21,800,000", "39.73%"],
    ]
    after = [
        ["股东名称", "持股数量（股）", "累计被质押数量（股）",
         "占其所持股份比例"],
        ["直接控股股东", "44,864,400", "26,830,000", "59.80%"],
        ["实际控制人", "10,008,279", "5,000,000", "49.96%"],
    ]
    rows = after_only_crossings([[release, after]])
    matched = [row for row in rows
               if row["release_actor_cell"] == "直接控股股东"
               and row["after_actor_cell"] == "直接控股股东"]
    assert len(matched) == 1
    assert matched[0]["released"] == 16_800_000
    assert matched[0]["crossed_tier"] == "80"


def test_one_share_rows_are_included_in_same_actor_sum() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "5,807,100", "19.09%"],
        ["直接控股股东", "1", "0.00%"],
        ["直接控股股东", "1", "0.00%"],
    ]
    after = [
        ["股东名称", "持股数量（股）", "累计被质押数量（股）",
         "占其所持股份比例"],
        ["直接控股股东", "30,421,897", "20,416,598", "67.11%"],
    ]
    rows = after_only_crossings([[release, after]])
    assert any(row["released"] == 5_807_102 for row in rows)


def test_trading_before_column_is_not_post_release_stock() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "14,500,000", "23.24%"],
    ]
    before_after = [
        ["股东名称", "持股数量（股）", "本次交易前目前累计质押股份数量（股）",
         "目前累计质押股份数量（股）", "占其所持股份比例"],
        ["直接控股股东", "62,389,317", "30,500,000", "16,000,000",
         "25.65%"],
    ]
    assert after_only_crossings([[release, before_after]]) == []


def test_extension_before_column_is_not_post_release_stock() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "9,030,000", "5.24%"],
    ]
    before_after = [
        ["股东名称", "持股数量（股）",
         "本次解除质押及质押股份延期购回前质押股份数量（股）",
         "本次解除质押及质押股份延期购回后质押股份数量（股）"],
        ["直接控股股东", "172,325,527", "84,319,159", "75,289,159"],
    ]
    assert after_only_crossings([[release, before_after]]) == []


def test_split_before_header_is_not_post_release_stock() -> None:
    release = [
        ["股东名称", "本次解除质押数量（股）", "占其所持股份比例"],
        ["直接控股股东", "39,000,000", "28.70%"],
    ]
    before_after = [
        ["股东名称", "持股数量（股）", "本次解除质", "本次解除"],
        [None, None, "押前质押股份数量", "质押后质押股份数量"],
        ["直接控股股东", "135,901,179", "84,000,000", "45,000,000"],
    ]
    assert after_only_crossings([[release, before_after]]) == []
