"""Import one verified market shard from a GitHub Actions artifact."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

import pyarrow.parquet as pq


API = "https://api.github.com/repos/solace47/trade/actions"


def _headers() -> dict[str, str]:
    result = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, check=True, timeout=10,
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return {
        "Authorization": f"Bearer {fields['password']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _api(path: str, headers: dict[str, str], method: str = "GET"):
    request = urllib.request.Request(API + path, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=30)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _byte_ranges(size: int, workers: int) -> list[tuple[int, int]]:
    if size <= 0 or workers <= 0:
        raise ValueError("Artifact size and worker count must be positive")
    width = (size + workers - 1) // workers
    return [(start, min(start + width, size) - 1)
            for start in range(0, size, width)]


def _download_ranges(location: str, target: Path, size: int) -> None:
    ranges = _byte_ranges(size, 4)
    parts = [target.with_name(f"{target.name}.part{index}")
             for index in range(len(ranges))]

    def one(index: int, start: int, end: int) -> None:
        request = urllib.request.Request(
            location, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as source:
            if source.status != 206 or source.headers.get("Content-Range") != \
                    f"bytes {start}-{end}/{size}":
                raise OSError("Artifact server did not honor a byte range")
            with parts[index].open("wb") as dest:
                shutil.copyfileobj(source, dest)
        if parts[index].stat().st_size != end - start + 1:
            raise OSError("Downloaded artifact part has an unexpected size")

    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [pool.submit(one, index, start, end)
                       for index, (start, end) in enumerate(ranges)]
            for future in as_completed(futures):
                future.result()
        with target.open("wb") as dest:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, dest)
        if target.stat().st_size != size:
            raise OSError("Combined artifact has an unexpected size")
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def _download(artifact_id: int, headers: dict[str, str], target: Path) -> None:
    temporary = target.with_suffix(".zip.tmp")
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                API + f"/artifacts/{artifact_id}/zip", headers=headers
            )
            opener = urllib.request.build_opener(_NoRedirect())
            try:
                opener.open(request, timeout=30)
                raise RuntimeError("Artifact API did not return a download location")
            except urllib.error.HTTPError as error:
                if error.code not in (301, 302, 303, 307, 308):
                    raise
                location = error.headers["Location"]
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(location, method="HEAD"), timeout=30
                ) as response:
                    size = int(response.headers.get("Content-Length", 0))
                    range_supported = response.headers.get("Accept-Ranges") == "bytes"
            except urllib.error.HTTPError:
                size, range_supported = 0, False
            if range_supported and size >= 8_000_000:
                _download_ranges(location, temporary, size)
            else:
                with urllib.request.urlopen(location, timeout=120) as source, \
                        temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            temporary.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def _extract(archive: Path, stage: Path) -> None:
    with zipfile.ZipFile(archive) as outer:
        members = [item for item in outer.infolist() if item.filename.endswith(".tar")]
        if len(members) != 1:
            raise ValueError("Expected exactly one shard tarball in artifact")
        tar_path = stage / "source.tar"
        with outer.open(members[0]) as source, tar_path.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    with tarfile.open(tar_path) as inner:
        for member in inner:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with inner.extractfile(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    tar_path.unlink()


def import_shard(run_id: int, shard: int, delete_remote: bool = False) -> dict:
    headers = _headers()
    with _api(f"/runs/{run_id}/artifacts", headers) as response:
        artifacts = json.load(response)["artifacts"]
    matches = [item for item in artifacts if item["name"] == f"market-shard-{shard}"]
    if len(matches) != 1:
        raise ValueError(f"Expected one artifact for shard {shard}; found {len(matches)}")
    artifact = matches[0]
    stage = Path("data/ci_shards") / f"run_{run_id}" / f"shard_{shard:02d}"
    stage.mkdir(parents=True, exist_ok=True)
    archive = stage / "artifact.zip"
    _download(artifact["id"], headers, archive)
    _extract(archive, stage)
    report = stage / f"data/research/shard_{shard}_result.json"
    lock = stage / f"data/research/shard_{shard}_source.lock"
    result = json.loads(report.read_text(encoding="utf-8"))
    revision = lock.read_text(encoding="utf-8").strip()
    for prior in Path("data/ci_shards").glob("run_*/shard_*/data/research/shard_*_source.lock"):
        if prior != lock and prior.read_text(encoding="utf-8").strip() != revision:
            raise ValueError("Market shards use different archive versions")
    copied = {}
    for kind in ("market_snapshots", "market_outcomes"):
        source = stage / "data/research" / kind
        files = sorted(source.glob("part_*.parquet"))
        if not files:
            raise ValueError(f"No {kind} files in shard {shard}")
        target_dir = Path("data/research") / f"{kind}_ci"
        target_dir.mkdir(parents=True, exist_ok=True)
        copied[kind] = 0
        for item in files:
            pq.ParquetFile(item)
            link = target_dir / f"shard_{shard:02d}_{item.name}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(item.resolve())
            copied[kind] += 1
    for source, destination in (
        (stage / "data/research/market_audit/issues.csv",
         Path("data/research/market_issues_ci") / f"shard_{shard:02d}.csv"),
        (stage / "data/research/market_audit/summary.json",
         Path("data/research/market_audit_ci") / f"shard_{shard:02d}.json"),
        (stage / "data/baostock/market_2020_2026/metadata/symbols.parquet",
         Path("data/research/market_symbols_ci") / f"shard_{shard:02d}.parquet"),
    ):
        if not source.exists():
            raise ValueError(f"Missing shard sidecar: {source.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(source.resolve())
    if delete_remote:
        with _api(f"/artifacts/{artifact['id']}", headers, method="DELETE") as response:
            if response.status != 204:
                raise RuntimeError("Could not remove imported remote artifact")
    local_lock = Path("data/hf/revision.lock")
    output = {
        "shard": shard, "symbols": result["selected_symbols"],
        "snapshots": result["snapshots"], "outcomes": result["outcomes"],
        "source_matches_local_lock": (
            local_lock.exists() and local_lock.read_text(encoding="utf-8").strip() == revision
        ),
        "partition_files": copied,
        "remote_artifact_removed": delete_remote,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--delete-remote", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_shard(args.run_id, args.shard, args.delete_remote),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
