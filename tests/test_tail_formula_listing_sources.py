import pytest

from trade_research.research_io import minute_sources


def test_minute_identity_uses_manifest_root_and_exchange():
    files = {'data/hf/pilot/data/stock_1m/SH/000001.parquet': 'first',
             'other/root/SZ/000001.parquet': 'second'}
    assert minute_sources(files) == {
        'sh.000001': 'data/hf/pilot/data/stock_1m/SH/000001.parquet',
        'sz.000001': 'other/root/SZ/000001.parquet'}


def test_ambiguous_stock_identity_cannot_silently_replace_source():
    with pytest.raises(AssertionError, match='Ambiguous'):
        minute_sources({'one/SH/600001.parquet': 'first',
                        'two/SH/600001.parquet': 'second'})
