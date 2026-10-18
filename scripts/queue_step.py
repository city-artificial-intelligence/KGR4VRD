"""Publish one frozen command specification to the existing Exact-OM dispatcher."""

import argparse
import fcntl
import hashlib
import json
import shlex
import sys
from pathlib import Path

from slurm_worker import write


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(spec_path, supervisor, exact_source, *, prepare_existing=False):
    spec_path, supervisor = Path(spec_path).resolve(), Path(supervisor).resolve()
    spec = json.loads(spec_path.read_text())
    directory = Path(spec["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    worker = directory / "worker.sh"
    if worker.exists():
        raise FileExistsError("Use a new attempt directory")
    helper = Path(__file__).with_name("slurm_worker.py").resolve()
    worker.write_text(
        "#!/bin/bash\nset -euo pipefail\nexec "
        + shlex.join([spec["python"], str(helper), str(spec_path)])
        + "\n"
    )
    resources = spec["resources"]
    argv = [
        "/usr/bin/srun",
        "--jobid=" + spec["allocation"],
        "--overlap",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=" + str(resources["cpus"]),
        "--mem=" + str(resources["memory_mb"]),
        "--gres=" + ("gpu:rtx2060:1" if resources["gpus"] else "none"),
        "--cpu-bind=none",
        "--unbuffered",
        "--job-name=" + spec["id"],
        "--chdir=" + spec["cwd"],
        "/bin/bash",
        str(worker),
    ]
    run = dict(
        id=spec["id"],
        resources=resources,
        depends_on=spec.get("depends_on", []),
        status_path=str(directory / "status.json"),
        exit_path=str(directory / "exit-code"),
        completion_path=str(directory / "completion.json"),
    )
    launch = dict(
        argv=argv,
        nonce=spec["nonce"],
        run=run,
        step_path=str(directory / "step.json"),
        launcher_log=str(directory / "worker.log"),
        tmux_socket=str(supervisor / "tmux.sock"),
        bindings=[dict(path=str(p), sha256=digest(p)) for p in (spec_path, worker, helper)]
        + spec.get("bindings", []),
    )
    batch = dict(id=spec["id"], resources=resources, depends_on=run["depends_on"], launch=launch)
    sys.path.insert(0, str(Path(exact_source).resolve()))
    from exact.experiments.dispatch import _validate

    _validate(batch, spec["allocation"])
    write(directory / "launch.json", batch)
    with (supervisor / "registry.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads((supervisor / "registry.json").read_text())
        identifiers = {r["id"] for r in registry["runs"] + registry["pending_batches"]}
        if batch["id"] in identifiers:
            existing = [r for r in registry["pending_batches"] if r["id"] == batch["id"]]
            if (
                not prepare_existing
                or any(r["id"] == batch["id"] for r in registry["runs"])
                or len(existing) != 1
                or existing[0].get("launch")
                or existing[0].get("enabled", True) is False
                or existing[0].get("depends_on", []) != batch["depends_on"]
                or existing[0]["resources"] != batch["resources"]
            ):
                raise ValueError("Logical ID already registered or incompatible preparation")
            # Attach the reviewed descriptor atomically; never remove an obligation
            # in a separate write or rewrite existing dependency links/history.
            existing[0].update(batch)
        elif prepare_existing:
            raise ValueError("No existing preparation obligation")
        else:
            registry["pending_batches"].append(batch)
        registry["remaining_work_status"] = "pending"
        write(supervisor / "registry.json", registry)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--exact-source", type=Path, required=True)
    parser.add_argument("--prepare-existing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            publish(
                args.spec,
                args.supervisor,
                args.exact_source,
                prepare_existing=args.prepare_existing,
            ),
            indent=2,
        )
    )
