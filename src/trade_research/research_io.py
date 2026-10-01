"""File receipts and explicit validation of the cleaned research runtime."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import subprocess

RUNTIME = Path('config/research-runtime.json')

def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


@lru_cache(maxsize=None)
def archived_sha(revision, file):
    content = subprocess.run(['git', 'show', revision + ':' + file], capture_output=True, check=True).stdout
    return hashlib.sha256(content).hexdigest()


def check_sources(sources):
    runtime = json.loads(RUNTIME.read_text())
    for file, digest in sources.items():
        if Path(file).exists() and sha(Path(file)) == digest:
            continue
        if file in runtime['historical_source_paths']:
            if archived_sha(runtime['archive_revision'], file) == digest:
                continue
        revision = runtime.get('additional_archived_sources', {}).get(file, {}).get(digest)
        assert revision is not None and archived_sha(revision, file) == digest, file


def check_runtime(*, committed=True):
    r = json.loads(RUNTIME.read_text())
    if r.get('schema_version',1) == 1:
        assert r['research_definition_changed'] is False and r['new_fits_performed'] == 0
    else:
        assert r['schema_version'] == 2
        for file,digest in r['prior_runtime_receipts'].items():
            assert sha(Path(file)) == digest, file
        if 'fixed_fit_runtime_sha256' in r:
            assert r['fixed_fit_runtime_sha256'] in r['prior_runtime_receipts'].values()
    assert r['new_2026_prices_read'] is False
    for file, digest in r['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    for file, digest in r['input_receipts'].items():
        assert sha(Path(file)) == digest, file
    if committed:
        text = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
                              capture_output=True, text=True, check=True).stdout
        assert sha(RUNTIME) in text, 'Commit the cleaned runtime before continuing research'
    return r
