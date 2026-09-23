"""Import one verified market shard from a GitHub Actions artifact."""

from __future__ import annotations

import argparse
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
            with urllib.request.urlopen(location, timeout=120) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
            return
        except (urllib.error.URLError, TimeoutError):
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
