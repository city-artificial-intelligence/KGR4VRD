"""Canonical identities and atomic, verifiable plain-file persistence."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: str | Path) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {path}: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def atomic_write(path: str | Path, writer: Any) -> None:
    """Write in the destination directory, then atomically publish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    atomic_write(path, lambda stream: stream.write(encoded))


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    def write(stream: Any) -> None:
        for record in records:
            stream.write(
                (
                    json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                ).encode()
            )

    atomic_write(path, write)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_file(base: Path, record: dict) -> Path:
    path = base / record["path"]
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def environment() -> dict:
    packages = {}
    for name in ("torch", "numpy", "Pillow", "rdflib", "owlrl", "torchvision", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def source_snapshot() -> dict:
    """Hash actual implementation bytes, including untracked edits; exclude research/data."""
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = sorted(package.rglob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()]
    files = {str(path.relative_to(root)): file_digest(path) for path in paths}
    contents = {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in paths}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "digest": canonical_digest(files),
        "files": files,
        "contents": contents,
        "git_commit": commit,
    }


def verify_run(run_directory: str | Path) -> dict:
    directory = Path(run_directory)
    run = read_json(directory / "run.json")
    if run.get("schema_version") != "kgr4vrd.run/1" or run.get("status") != "completed":
        raise ValueError(f"Run is not complete: {directory}")
    if canonical_digest(run["identity"]) != run["run_id"]:
        raise ValueError("Run identity mismatch")
    required = {
        "resolved-config.json",
        "environment.json",
        "source.json",
        "epochs.jsonl",
        "events.jsonl",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
        "selected.json",
    }
    if not required.issubset(run["outputs"]):
        raise ValueError("Run manifest omits required outputs")
    for name, digest in run["outputs"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Run output checksum mismatch: {name}")
    selection = read_json(directory / "selected.json")
    if (
        selection["run_id"] != run["run_id"]
        or selection["identity"] != run["identity"]
        or selection["metric"] != "recall_image_mean_at_50"
        or selection["tie_rule"] != "earliest"
    ):
        raise ValueError("Invalid selection identity or policy")
    if file_digest(directory / selection["checkpoint"]) != selection["checkpoint_digest"]:
        raise ValueError("Selected checkpoint checksum mismatch")
    epochs = read_jsonl(directory / "epochs.jsonl")
    if len(epochs) != run["completed_epochs"]:
        raise ValueError("Missing completed epoch records")
    best = max(epochs, key=lambda row: row["selection_score"])
    if best["epoch"] != selection["epoch"] or best["selection_score"] != selection["value"]:
        raise ValueError("Selection is not the earliest best validation epoch")
    return {"run_id": run["run_id"], "status": "verified", "files": len(run["outputs"])}
