import pandas as pd

from trade_research.annual_cash_direct_inputs import _two_cell_coverage


def test_two_cell_audit_excludes_unmatched_cash_group() -> None:
    rows = [
        {"date": "2024-05-15", "board": "sh_main", "size_bucket": 1,
         "momentum_bucket": 1, "cash_group": group,
         "code": f"sh.{number:06d}", "float_mv": size,
         "avg20_amount": amount}
        for number, (group, size, amount) in enumerate([
            ("cash_supported", 120.0, 90.0),
            ("low_cash_conversion", 100.0, 60.0),
        ])
    ]
    rows.append({**rows[0], "date": "2024-05-16", "code": "sh.600002"})
    audit = _two_cell_coverage(pd.DataFrame(rows))["2024"]
    assert audit["source_strata"] == 2
    assert audit["two_cell_strata"] == 1
    assert audit["source_stock_days"] == 3
    assert audit["two_cell_stock_days"] == 2
    assert audit["median_supported_low_ratio"] == {
        "float_mv": 1.2, "avg20_amount": 1.5,
    }
