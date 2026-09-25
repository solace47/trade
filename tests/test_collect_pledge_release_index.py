from scripts.collect_pledge_release_index import candidate_title


def test_conservative_controller_release_title() -> None:
    assert candidate_title("关于控股股东部分股份解除质押的公告")
    assert candidate_title("关于控股股东、实际控制人股份解除质押的公告")
    assert not candidate_title("关于控股股东之一致行动人股份解除质押的公告")
    assert not candidate_title("关于控股股东股份解除质押及再质押的公告")
    assert not candidate_title("关于控股股东可交换公司债券解除质押的公告")
    assert not candidate_title("关于控股股东拟办理股份解除质押的公告")
