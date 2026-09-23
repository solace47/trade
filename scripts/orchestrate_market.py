"""Import completed market artifacts and dispatch the remaining bounded batches.

This process keeps only a few hosted artifacts at a time. Its ignored state
file makes a restart safe after an interrupted network request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error

from import_ci_artifact import _api, _headers, import_shard


STATE = Path("data/research/orchestration.json")


def _read(path: str, headers: dict[str, str]) -> dict:
    with _api(path, headers) as response:
        return json.load(response)


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(STATE)


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(stamp, message, flush=True)


def _dispatch(headers: dict[str, str], first: int, count: int) -> int:
    before = _read("/workflows/market-shard.yml/runs?per_page=1", headers)
    latest_id = before["workflow_runs"][0]["id"]
    payload = {"ref": "main", "inputs": {
        "first_shard": str(first), "max_shards": str(count), "limit_symbols": "0",
    }}
    request = json.dumps(payload).encode()
    with _api_dispatch(headers, request) as response:
        if response.status != 204:
            raise RuntimeError(f"Dispatch returned {response.status}")
    for _ in range(24):
        time.sleep(5)
        recent = _read("/workflows/market-shard.yml/runs?per_page=5", headers)
        matches = [run for run in recent["workflow_runs"]
                   if run["id"] > latest_id and run["event"] == "workflow_dispatch"]
        if matches:
            return max(run["id"] for run in matches)
    raise TimeoutError("Dispatched market workflow was not listed within two minutes")


def _api_dispatch(headers: dict[str, str], payload: bytes):
    import urllib.request

    request = urllib.request.Request(
        "https://api.github.com/repos/solace47/trade/actions/"
        "workflows/market-shard.yml/dispatches",
        data=payload, headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=30)


def run(initial_run_id: int, batch_size: int = 4, total_shards: int = 20) -> None:
    if not 1 <= batch_size <= total_shards <= 20:
        raise ValueError("Invalid batch configuration")
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {
            "active_run_id": initial_run_id, "active_shards": list(range(batch_size)),
            "next_shard": batch_size, "imported": [], "attempts": {},
        }
        _save(state)
    headers = _headers()
    while True:
        active_id = state["active_run_id"]
        if active_id is None:
            if state["next_shard"] >= total_shards:
                _log(f"All {total_shards} shards imported")
                return
            first = state["next_shard"]
            count = min(batch_size, total_shards - first)
            active_id = _dispatch(headers, first, count)
            state.update(active_run_id=active_id,
                         active_shards=list(range(first, first + count)),
                         next_shard=first + count)
            _save(state)
            _log(f"Dispatched run {active_id} for shards {first}-{first + count - 1}")
        try:
            run_info = _read(f"/runs/{active_id}", headers)
            artifacts = _read(f"/runs/{active_id}/artifacts?per_page=100", headers)
        except (urllib.error.URLError, TimeoutError) as error:
            _log(f"Transient API error: {type(error).__name__}")
            time.sleep(30)
            continue
        available = {item["name"] for item in artifacts["artifacts"]}
        for shard in state["active_shards"]:
            if shard in state["imported"] or f"market-shard-{shard}" not in available:
                continue
            try:
                result = import_shard(active_id, shard, delete_remote=True)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                attempts = state["attempts"].get(str(shard), 0) + 1
                state["attempts"][str(shard)] = attempts
                _save(state)
                _log(f"Import retry {attempts} for shard {shard}: {type(error).__name__}")
                if attempts >= 10:
                    raise
                continue
            if not result["source_matches_local_lock"]:
                raise ValueError(f"Source version differs from local lock for shard {shard}")
            state["imported"].append(shard)
            state["imported"].sort()
            _save(state)
            _log(f"Imported shard {shard}: {result['symbols']} symbols, "
                 f"{result['snapshots']} snapshots")
        done = all(shard in state["imported"] for shard in state["active_shards"])
        if run_info["status"] == "completed":
            if run_info["conclusion"] != "success" or not done:
                raise RuntimeError(f"Run {active_id} ended {run_info['conclusion']}; "
                                   f"imported {sum(s in state['imported'] for s in state['active_shards'])} "
                                   f"of {len(state['active_shards'])} shards")
            state["active_run_id"] = None
            state["active_shards"] = []
            _save(state)
            continue
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-shards", type=int, default=20)
    args = parser.parse_args()
    run(args.initial_run_id, args.batch_size, args.total_shards)


if __name__ == "__main__":
    main()
