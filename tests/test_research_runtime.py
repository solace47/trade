import hashlib
import json

import pytest

from trade_research import research_io


@pytest.mark.parametrize('file_name', ['active.py', 'features.parquet'])
def test_runtime_rejects_changed_code_or_existing_input(tmp_path, monkeypatch, file_name):
    code = tmp_path / 'active.py'
    inputs = tmp_path / 'features.parquet'
    code.write_bytes(b'fixed source')
    inputs.write_bytes(b'fixed input')
    runtime = tmp_path / 'runtime.json'
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    runtime.write_text(json.dumps(dict(
        research_definition_changed=False, new_fits_performed=0, new_2026_prices_read=False,
        source_hashes={str(code): digest(code)}, input_receipts={str(inputs): digest(inputs)})))
    monkeypatch.setattr(research_io, 'RUNTIME', runtime)
    research_io.check_runtime(committed=False)
    changed = tmp_path / file_name
    changed.write_bytes(b'changed')
    with pytest.raises(AssertionError, match=file_name):
        research_io.check_runtime(committed=False)
