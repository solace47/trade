"""Independent SQL ranks, denominators, matched cells and reported peer statistics."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/winner_peer_timing")
protocol = json.loads(Path("config/winner_peer_timing_protocol.json").read_text())
report = json.loads((root / "analysis_report.json").read_text())
for name, digest in report["outputs_sha256"].items():
    assert sha(root / (name + ".parquet")) == digest
metrics = ["up", "down", "balance", "reference_gain"]
stored_ranks = pd.read_parquet(root / "ranks.parquet").set_index(["date", "code"]).sort_index()
stored_daily = pd.read_parquet(root / "band_daily.parquet")
stored_match = pd.read_parquet(root / "matched_daily.parquet")
c = duckdb.connect()
c.execute("SET threads=4")
c.execute("SET memory_limit='6GB'")
c.execute("""CREATE TABLE f AS SELECT f.*,l.known_label,l.cohort,
    CASE WHEN l.known_label THEN (round(l.next_close*100)::BIGINT*100>=round(l.next_preclose*100)::BIGINT*105)::DOUBLE END AS up,
    CASE WHEN l.known_label THEN (round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95)::DOUBLE END AS down,
    l.next_gain AS reference_gain,up-down AS balance,
    CASE WHEN return_tail29>.005 THEN 'rising' WHEN return_tail29<-.005 THEN 'falling' ELSE 'flat' END AS own_tail_band
    FROM read_parquet('data/research/winner_peer_timing/features.parquet') f
    JOIN read_parquet('data/research/next_day_winner/labels.parquet') l USING(date,code)""")


def compare(left, right, keys, values):
    left = left.set_index(keys).sort_index()
    right = right.set_index(keys).sort_index()
    assert left.index.equals(right.index), (len(left), len(right))
    a, b = left[values].to_numpy(dtype=float), right[values].to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)


aggregation = '''count(*) AS rows,sum(known_label::INT) AS known,
    sum((NOT known_label)::INT) AS unknown,sum(up) AS winners,sum(down) AS losers,
    avg(up) AS up,avg(down) AS down,avg(balance) AS balance,avg(reference_gain) AS reference_gain'''
counts = ["rows", "known", "unknown", "winners", "losers"]
independent_daily = []
independent_matched = []
rank_values = 0
portrait_cells = 0
for feature in protocol["features"]:
    c.execute(f'''CREATE OR REPLACE TABLE r AS WITH ranked AS (
        SELECT *,CASE WHEN "{feature}" IS NOT NULL THEN
          (rank() OVER(PARTITION BY date,board ORDER BY "{feature}" NULLS LAST)
            +(count(*) OVER(PARTITION BY date,board,"{feature}")-1)/2.0)
          /count("{feature}") OVER(PARTITION BY date,board) END AS feature_rank FROM f)
        SELECT *,CASE WHEN feature_rank IS NULL THEN 'missing' WHEN feature_rank<=.2 THEN 'low20'
          WHEN feature_rank>=.8 THEN 'high20' ELSE 'middle60' END AS band FROM ranked''')
    ranks = c.sql("SELECT date,code,feature_rank FROM r").df().set_index(["date", "code"]).sort_index()
    np.testing.assert_allclose(ranks.feature_rank, stored_ranks[feature], atol=1e-13, rtol=1e-13, equal_nan=True)
    rank_values += len(ranks)
    for scope, condition in (("all", "true"), ("necessary", "necessary_tradeable")):
        daily = c.sql(f"SELECT date,board,band,{aggregation} FROM r WHERE {condition} GROUP BY date,board,band").df()
        baseline = c.sql(f'''SELECT date,board,avg(up) AS up_baseline,avg(down) AS down_baseline,
            avg(balance) AS balance_baseline,avg(reference_gain) AS reference_gain_baseline
            FROM r WHERE {condition} GROUP BY date,board''').df()
        daily = daily.merge(baseline, on=["date", "board"], validate="many_to_one")
        daily["feature"], daily["scope"] = feature, scope
        for metric in metrics:
            daily[metric + "_difference"] = daily[metric] - daily[metric + "_baseline"]
        reference = stored_daily.loc[stored_daily.feature.eq(feature) & stored_daily.scope.eq(scope)]
        compare(daily, reference, ["date", "board", "band"],
            counts + metrics + [v + "_difference" for v in metrics])
        independent_daily.append(daily)
        portrait = c.sql(f'''SELECT half,board,cohort,count(*) AS rows,count("{feature}") AS observed,
            avg("{feature}") AS mean_value,median("{feature}") AS median_value,avg(feature_rank) AS mean_rank
            FROM r WHERE {condition} GROUP BY half,board,cohort''').df()
        saved_portrait = pd.read_parquet(root / "portraits.parquet")
        saved_portrait = saved_portrait.loc[saved_portrait.feature.eq(feature) & saved_portrait.scope.eq(scope)]
        compare(portrait, saved_portrait, ["half", "board", "cohort"],
            ["rows", "observed", "mean_value", "median_value", "mean_rank"])
        portrait_cells += len(portrait)
    matched = c.sql('''WITH bins AS (
        SELECT *,floor(return_1449/.02) AS day_bin,floor(return20_prior_adjusted/.1) AS prior_bin,
            floor(log2(price_1449/5)) AS price_bin,floor(log2(amount_1449/3e7)) AS amount_bin,
            floor((high_1449-low_1449)/preclose/.02) AS range_bin,floor(return_tail29/.01) AS tail_bin
        FROM r WHERE necessary_tradeable AND known_label AND return20_prior_adjusted IS NOT NULL AND feature_rank IS NOT NULL
    ), cells AS (
        SELECT date,board,day_bin,prior_bin,price_bin,amount_bin,range_bin,tail_bin,
            count(*) FILTER(WHERE up=1) AS n1,count(*) FILTER(WHERE up=0) AS n0,
            avg(feature_rank) FILTER(WHERE up=1)-avg(feature_rank) FILTER(WHERE up=0) AS difference
        FROM bins GROUP BY date,board,day_bin,prior_bin,price_bin,amount_bin,range_bin,tail_bin
    ) SELECT date,board,sum(n1) AS weight,sum(difference*n1) AS weighted_difference,
        sum(difference*n1)/sum(n1) AS rank_difference FROM cells WHERE n1>0 AND n0>0 GROUP BY date,board''').df()
    compare(matched, stored_match.loc[stored_match.feature.eq(feature)], ["date", "board"],
        ["weight", "weighted_difference", "rank_difference"])
    matched["feature"] = feature
    independent_matched.append(matched)
    if feature == "peer_tail_return":
        interaction = c.sql(f'''SELECT date,board,own_tail_band,band,{aggregation}
            FROM r WHERE necessary_tradeable GROUP BY date,board,own_tail_band,band''').df()
        baseline = c.sql('''SELECT date,board,own_tail_band,avg(up) AS up_baseline,avg(down) AS down_baseline,
            avg(balance) AS balance_baseline,avg(reference_gain) AS reference_gain_baseline
            FROM r WHERE necessary_tradeable GROUP BY date,board,own_tail_band''').df()
        interaction = interaction.merge(baseline, on=["date", "board", "own_tail_band"], validate="many_to_one")
        for metric in metrics:
            interaction[metric + "_difference"] = interaction[metric] - interaction[metric + "_baseline"]
        compare(interaction, pd.read_parquet(root / "interaction_daily.parquet"),
            ["date", "board", "own_tail_band", "band"], counts + metrics + [v + "_difference" for v in metrics])
    print(json.dumps({"result_feature_checked": feature}), flush=True)


def independent_interval(series):
    if not len(series):
        return None
    weeks = pd.to_datetime(series.index).to_period("W-SUN").astype(str)
    blocks = pd.DataFrame({"week": weeks, "value": series.to_numpy()}).groupby("week").value.agg(["sum", "count"])
    if len(blocks) < 2:
        return None
    rng = np.random.default_rng(20260926)
    draws = rng.integers(len(blocks), size=(10000, len(blocks)))
    sums = np.take(blocks["sum"].to_numpy(), draws).sum(axis=1)
    n = np.take(blocks["count"].to_numpy(), draws).sum(axis=1)
    return np.percentile(sums / n, [2.5, 97.5])


summary_cells = 0
for file, frame, keys in (("band_summary", pd.concat(independent_daily, ignore_index=True), ["feature", "scope", "board", "band"]),
                         ("interaction_summary", interaction, ["board", "own_tail_band", "band"])):
    frame["half"] = frame.date.str[:4] + np.where(frame.date.str[5:7].le("06"), "H1", "H2")
    stored = pd.read_parquet(root / (file + ".parquet")).set_index(["half", *keys])
    for key, part in frame.groupby(["half", *keys]):
        row = stored.loc[key]
        assert row["days"] == part.known.gt(0).sum()
        for name in counts:
            assert row[name] == int(part[name].sum())
        for metric in metrics:
            for name in (metric, metric + "_difference"):
                values = part.set_index("date")[name].dropna()
                if len(values):
                    assert abs(values.mean() - row[name]) < 1e-12
                else:
                    assert pd.isna(row[name])
                interval = independent_interval(values)
                actual = row[name + "_week_ci"]
                if interval is None:
                    assert actual is None
                else:
                    np.testing.assert_allclose(interval, actual, atol=1e-12, rtol=1e-12)
        summary_cells += 1
matched_frame = pd.concat(independent_matched, ignore_index=True)
matched_frame["half"] = matched_frame.date.str[:4] + np.where(matched_frame.date.str[5:7].le("06"), "H1", "H2")
all_winners = c.sql('SELECT half,board,sum(up) AS n FROM f WHERE necessary_tradeable GROUP BY half,board').df().set_index(["half", "board"])
matched_summary = pd.read_parquet(root / "matched_summary.parquet").set_index(["feature", "half", "board"])
for key, part in matched_frame.groupby(["feature", "half", "board"]):
    row = matched_summary.loc[key]
    total = all_winners.loc[key[1:], "n"]
    assert row["days"] == len(part) and row.matched_winners == part.weight.sum() and row.all_winners == total
    assert abs(row.coverage - part.weight.sum() / total) < 1e-12
    assert abs(row.rank_difference - part.rank_difference.mean()) < 1e-12
    np.testing.assert_allclose(row.rank_difference_week_ci, independent_interval(part.set_index("date").rank_difference), atol=1e-12, rtol=1e-12)
baseline = c.sql('''SELECT half,board,avg(up) AS up,avg(down) AS down,avg(balance) AS balance,
    avg(reference_gain) AS reference_gain FROM (
        SELECT half,date,board,avg(up) AS up,avg(down) AS down,avg(balance) AS balance,avg(reference_gain) AS reference_gain
        FROM f WHERE necessary_tradeable GROUP BY half,date,board) GROUP BY half,board''').df()
compare(baseline, pd.DataFrame(report["baseline"]).query("scope=='necessary'"), ["half", "board"], metrics)
result = {"rank_values_checked": rank_values, "band_daily_cells": len(stored_daily), "portrait_cells": portrait_cells,
    "matched_daily_cells": sum(len(p) for p in independent_matched), "interaction_daily_cells": len(interaction),
    "summary_cells_and_block_intervals_checked": summary_cells + len(matched_summary),
    "analysis_report_sha256": sha(root / "analysis_report.json"), "new_2026_prices_read": False}
save_json(root / "independent_result_checks.json", result)
print(json.dumps(result, indent=2))
