"""Acquire official PhysioNet PSV files for research over HTTPS."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import tempfile
import time
import threading
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
import requests

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
INDEX = "https://physionet.org/content/challenge-2019/1.0.0/"
SOURCES = {"A": "training_setA", "B": "training_setB"}
_PATIENT_NAME = re.compile(r"p[0-9]+\.psv\Z", re.IGNORECASE)
_OFFICIAL_HOSTS = {"physionet.org", "www.physionet.org"}
S3_PREFIX = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/"
_session_local = threading.local()


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def _request(url: str, *, timeout: int = 30, retries: int = 4,
             retry_delay: float = 0.5):
    """Open an official URL, retrying bounded transient transport/server errors."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_HOSTS:
        raise ValueError(f"unexpected PhysioNet URL: {url}")
    request = Request(url, headers={"User-Agent": "icu-sepsis-research-audit/0.1"})
    for attempt in range(retries + 1):
        try:
            response = urlopen(request, timeout=timeout)
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname not in _OFFICIAL_HOSTS:
                response.close()
                raise ValueError(f"unexpected download redirect: {response.geturl()}")
            return response
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt >= retries:
                raise
            exc.close()
            delay = exc.headers.get("Retry-After") if exc.headers else None
            try:
                wait = min(float(delay), 30.0) if delay is not None else retry_delay * (2 ** attempt)
            except ValueError:
                wait = retry_delay * (2 ** attempt)
        except (URLError, TimeoutError, OSError) as exc:
            if attempt >= retries:
                raise
            wait = retry_delay * (2 ** attempt)
        time.sleep(wait)
    raise AssertionError("unreachable")


class _RequestsStream:
    """Small read/context adapter for a streamed requests.Response."""
    def __init__(self, response: requests.Response):
        self.response = response
        self.buffer = bytearray()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            self.buffer.extend(b"".join(self.response.raw.stream(1024 * 1024, decode_content=True)))
            data = bytes(self.buffer)
            self.buffer.clear()
            return data
        while len(self.buffer) < size:
            try:
                self.buffer.extend(next(self.response.raw.stream(size, decode_content=True)))
            except StopIteration:
                break
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data

    def close(self) -> None:
        self.response.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _s3_url(canonical_url: str) -> str:
    parsed = urlparse(canonical_url)
    base_path = urlparse(BASE).path
    if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_HOSTS or not parsed.path.startswith(base_path):
        raise ValueError(f"unexpected canonical PhysioNet URL: {canonical_url}")
    return S3_PREFIX + parsed.path[len(base_path):]


def _s3_request(url: str, *, timeout: int = 30, retries: int = 4,
                retry_delay: float = 0.5) -> _RequestsStream:
    parsed = urlparse(url)
    prefix = urlparse(S3_PREFIX)
    if parsed.scheme != "https" or parsed.hostname != prefix.hostname or not parsed.path.startswith(prefix.path):
        raise ValueError(f"unexpected official S3 mirror URL: {url}")
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        _session_local.session = session
    for attempt in range(retries + 1):
        response = None
        try:
            response = session.get(url, timeout=timeout, stream=True,
                                   headers={"User-Agent": "icu-sepsis-research-audit/0.1"})
            final = urlparse(response.url)
            if final.scheme != "https" or final.hostname != prefix.hostname or not final.path.startswith(prefix.path):
                response.close()
                raise ValueError(f"unexpected mirror redirect: {response.url}")
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt >= retries:
                    response.raise_for_status()
                delay = response.headers.get("Retry-After")
                try:
                    wait = min(float(delay), 30.0) if delay is not None else retry_delay * (2 ** attempt)
                except ValueError:
                    wait = retry_delay * (2 ** attempt)
                response.close()
                time.sleep(wait)
                continue
            response.raise_for_status()
            return _RequestsStream(response)
        except requests.RequestException:
            if response is not None:
                response.close()
            if attempt >= retries:
                raise
            time.sleep(retry_delay * (2 ** attempt))
    raise AssertionError("unreachable")


def _listing(source_dir: str, *, timeout: int = 30, retries: int = 4) -> tuple[str, list[str]]:
    if source_dir not in SOURCES.values():
        raise ValueError(f"unknown PhysioNet source directory: {source_dir}")
    url = urljoin(BASE, source_dir + "/")
    with _request(url, timeout=timeout, retries=retries) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _Links()
    parser.feed(html)
    names: set[str] = set()
    for href in parser.hrefs:
        candidate = unquote(urlparse(href).path.rstrip("/").split("/")[-1])
        if _PATIENT_NAME.fullmatch(candidate):
            names.add(candidate)
    if not names:
        raise RuntimeError(f"official PhysioNet index returned no PSV patient links: {url}")
    return url, sorted(names, key=str.lower)


def _file_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as existing:
        for block in iter(lambda: existing.read(1024 * 1024), b""):
            digest.update(block)
    return path.stat().st_size, digest.hexdigest()


def _download_one(source: str, listing_url: str, source_root: Path,
                  root: Path, name: str, timeout: int, retries: int = 4,
                  prior: dict | None = None, mirror: str = "physionet") -> dict:
    if not _PATIENT_NAME.fullmatch(name):
        raise ValueError(f"invalid patient filename: {name}")
    canonical_url = urljoin(listing_url, name)
    parsed = urlparse(canonical_url)
    expected_path = urlparse(listing_url).path.rstrip("/") + "/" + name
    if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_HOSTS or parsed.path != expected_path:
        raise ValueError(f"unexpected download URL: {canonical_url}")
    destination = source_root / name
    relative = destination.relative_to(root).as_posix()
    # Local files are reusable only when a persisted official manifest already
    # binds this exact source/path/URL to their matching digest.
    if mirror not in {"physionet", "s3"}:
        raise ValueError("mirror must be physionet or s3")
    download_url = canonical_url if mirror == "physionet" else _s3_url(canonical_url)
    if destination.is_file() and prior and prior.get("path") == relative and prior.get("source") == source and prior.get("url") == canonical_url:
        size, digest = _file_hash(destination)
        if size == prior.get("bytes") and digest == prior.get("sha256"):
            return prior
    part_path: Path | None = None
    try:
        request = _request if mirror == "physionet" else _s3_request
        with request(download_url, timeout=timeout, retries=retries) as response:
            with tempfile.NamedTemporaryFile(dir=source_root, prefix=f".{name}.",
                                             suffix=".part", delete=False) as tmp:
                part_path = Path(tmp.name)
                digest = hashlib.sha256()
                size = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    tmp.write(block)
                    digest.update(block)
                    size += len(block)
        if size == 0:
            raise IOError(f"empty response for {download_url}")
        part_path.replace(destination)
    except Exception:
        if part_path is not None:
            part_path.unlink(missing_ok=True)
        raise
    return {"source": source, "patient_id": Path(name).stem,
            "path": relative, "bytes": size, "sha256": digest.hexdigest(),
            "url": canonical_url, "download_url": download_url}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        tmp.write(json.dumps(manifest, indent=2))
        tmp.flush()
        # Ensure the checkpoint survives process termination after replace.
        import os
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    temporary.replace(path)


def download_physionet(output_dir: str | Path = "data/raw", *,
                       sources: tuple[str, ...] = ("A", "B"),
                       limit: int | None = None, timeout: int = 60,
                       workers: int = 8,
                       progress_callback: Callable[[int, int, dict], None] | None = None,
                       retries: int = 4, mirror: str = "physionet") -> dict:
    """Fetch selected official A/B PSV files; `limit` applies per source.

    Verified downloads are atomically renamed and a manifest checkpoint is
    atomically written in bounded batches, on errors and on interruption.
    Existing files without a matching persisted official checksum are refetched.
    """
    normalized = tuple(str(source).upper() for source in sources)
    if not normalized or any(source not in SOURCES for source in normalized):
        raise ValueError("sources must contain A and/or B")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sources must not repeat")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if workers < 1 or retries < 0 or timeout < 1:
        raise ValueError("workers/timeout must be positive and retries nonnegative")
    if mirror not in {"physionet", "s3"}:
        raise ValueError("mirror must be physionet or s3")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    prior_by_path = {e.get("path"): e for e in old.get("files", [])
                     if isinstance(e, dict)} if old.get("source_type") == "physionet_public_training_data" and old.get("official_file_base") == BASE else {}
    entries_by_path = dict(prior_by_path)
    errors: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    selected_paths: set[str] = set()
    manifest = {
        "report_type": "research_dataset_acquisition_manifest",
        "source_type": "physionet_public_training_data",
        "official_dataset_page": INDEX, "official_file_base": BASE,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0 (per official PhysioNet page)",
        "sources": list(normalized), "requested_limit_per_source": limit,
        "mirror": mirror,
        "concurrent_workers": workers, "source_file_counts": source_counts,
        "is_full_selected_sources": limit is None, "complete": False,
        # Preserve prior verified provenance even if the next index request is
        # interrupted before this run can make any progress.
        "files": [entries_by_path[p] for p in sorted(entries_by_path)], "errors": {},
        "note": "Research data acquisition only; this manifest does not report clinical performance.",
    }
    _write_manifest(manifest_path, manifest)
    total = 0
    for source in normalized:
        directory = SOURCES[source]
        try:
            listing_url, names = _listing(directory, timeout=timeout, retries=retries)
        except Exception as exc:
            errors[f"{directory}/<index>"] = f"{type(exc).__name__}: {exc}"
            manifest.update(errors=dict(errors), files=[entries_by_path[p] for p in sorted(entries_by_path)],
                            complete=False)
            _write_manifest(manifest_path, manifest)
            raise
        if limit is not None:
            names = names[:limit]
        source_root = root / directory
        source_root.mkdir(parents=True, exist_ok=True)
        source_counts[source] = len(names)
        jobs = []
        future_paths = {}
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            for name in names:
                path = f"{directory}/{name}"
                selected_paths.add(path)
                future = executor.submit(_download_one, source, listing_url,
                                         source_root, root, name, timeout, retries,
                                         prior_by_path.get(path), mirror)
                jobs.append((path, future))
                future_paths[future] = path
            total += len(jobs)
            completed = 0
            for future in as_completed(future_paths):
                path = future_paths[future]
                failed = False
                try:
                    entry = future.result()
                    entries_by_path[path] = entry
                    errors.pop(path, None)
                except Exception as exc:
                    errors[path] = f"{type(exc).__name__}: {exc}"
                    failed = True
                completed += 1
                # A full sorted manifest each file is quadratic for the ~40k
                # challenge set. Checkpoint every 100, every error, and source end.
                if completed % 100 == 0 or failed or completed == len(jobs):
                    manifest.update(files=[entries_by_path[p] for p in sorted(entries_by_path)],
                                    errors=dict(errors), complete=False,
                                    downloaded_at_utc=datetime.now(timezone.utc).isoformat())
                    _write_manifest(manifest_path, manifest)
                if progress_callback:
                    progress_callback(completed, len(jobs), {"path": path, "error": errors.get(path)})
        except KeyboardInterrupt:
            for future in future_paths:
                future.cancel()
            manifest.update(files=[entries_by_path[p] for p in sorted(entries_by_path)],
                            errors=dict(errors), complete=False,
                            downloaded_at_utc=datetime.now(timezone.utc).isoformat())
            _write_manifest(manifest_path, manifest)
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    manifest.update(files=[entries_by_path[p] for p in sorted(entries_by_path)],
                    errors=dict(errors), complete=not errors,
                    is_full_selected_sources=limit is None,
                    downloaded_at_utc=datetime.now(timezone.utc).isoformat())
    _write_manifest(manifest_path, manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} official file download(s) failed; see {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw", help="Raw-data root (ignored by source control)")
    parser.add_argument("--sources", nargs="+", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--limit", type=int, help="Maximum patients per source for a smoke subset")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--mirror", choices=("physionet", "s3"), default="physionet")
    args = parser.parse_args()
    started = time.monotonic()
    def progress(done: int, total: int, result: dict) -> None:
        if done % 250 == 0 or done == total:
            suffix = f"; last error: {result['error']}" if result.get("error") else ""
            print(f"{done}/{total} files in source complete{suffix}", flush=True)
    manifest = download_physionet(args.output, sources=tuple(args.sources), limit=args.limit,
                                  timeout=args.timeout, workers=args.workers,
                                  retries=args.retries, mirror=args.mirror,
                                  progress_callback=progress)
    print(json.dumps({"manifest": str((Path(args.output) / "download_manifest.json").resolve()),
                      "source_file_counts": manifest["source_file_counts"],
                      "limited_subset": args.limit is not None,
                      "elapsed_seconds": round(time.monotonic() - started, 2)}, indent=2))


if __name__ == "__main__":
    main()
