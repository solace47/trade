from scripts.propose_pledge_after_only import after_only_crossings


def test_after_only_table_can_propose_direct_tier_crossing() -> None:
    release = [
        ["股东名称", "本次解除质押数量（万股）", "占其所持股份比例"],
        ["锦隆能源", "19,750.00", "21.63%"],
        ["合计", "20,925.00", "22.91%"],
    ]
    after = [
        ["股东名称", "持股数量（万股）", "累计被质押数量（万股）",
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
