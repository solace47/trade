"""Check annual-report source keys and restart-safe checkpoints."""

import json

import pandas as pd

from scripts import collect_annual_cash_quality as collector


class FakeResult:
    error_code = "0"
    error_msg = ""

    def __init__(self, row):
        self.fields = list(row)
        self.row = row
        self.read = False

    def next(self):
        if self.read:
            return False
        self.read = True
        return True

    def get_row_data(self):
        return list(self.row.values())


def test_annual_cash_collection_resumes_without_duplicate_queries(
        tmp_path, monkeypatch) -> None:
    universe = tmp_path / "universe.parquet"
    pd.DataFrame({"code": ["sh.600004", "sh.600006"],
                  "trade_date": ["2024-05-01", "2024-05-01"]}).to_parquet(
        universe, index=False)
    calls = []

    def query_profit_data(*, code, year, quarter):
        calls.append(("profit", code, year, quarter))
        return FakeResult({"code": code, "pubDate": f"{year+1}-04-30",
                           "statDate": f"{year}-12-31", "netProfit": "100"})

    def query_cash_flow_data(*, code, year, quarter):
        calls.append(("cash", code, year, quarter))
        return FakeResult({"code": code, "pubDate": f"{year+1}-04-30",
                           "statDate": f"{year}-12-31", "CFOToNP": "1.1"})

    monkeypatch.setattr(collector.bs, "query_profit_data", query_profit_data)
    monkeypatch.setattr(collector.bs, "query_cash_flow_data", query_cash_flow_data)
    monkeypatch.setattr(collector.bs, "login", lambda: type("Login", (), {
        "error_code": "0", "error_msg": ""})())
    monkeypatch.setattr(collector.bs, "logout", lambda: None)
    output = tmp_path / "annual"
    first = collector.collect(universe, output, shard=0, shards=1,
                              max_codes=1)
    second = collector.collect(universe, output, shard=0, shards=1)
    third = collector.collect(universe, output, shard=0, shards=1)
    assert first["fetched"] == 2
    assert second["fetched"] == 2
    assert third["fetched"] == 0
    assert len(calls) == 8
    assert all(quarter == 4 for _, _, _, quarter in calls)
    records = [json.loads(line) for line in
               (output / "annual_cash_00_of_01.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert {(item["code"], item["year"]) for item in records} == {
        (code, year) for code in ("sh.600004", "sh.600006")
        for year in (2023, 2024)}
