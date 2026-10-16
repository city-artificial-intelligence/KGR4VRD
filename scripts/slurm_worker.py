"""Run a frozen command list and publish Exact-OM-compatible Slurm receipts."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    directory = Path(spec["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "step.json").exists():
        raise RuntimeError("Attempt already started; inspect owner and create a new attempt")
    assert os.environ["SLURM_JOB_ID"] == spec["allocation"]
    assert os.environ["SLURM_STEP_ID"].isdigit()
    step = spec["allocation"] + "." + os.environ["SLURM_STEP_ID"]
    started = time.time()
    receipt = dict(
        step_id=step,
        dispatch_nonce=spec["nonce"],
        pid=os.getpid(),
        started_epoch=started,
        resources=spec["resources"],
    )
    write(directory / "step.json", receipt)
    write(directory / "status.json", dict(receipt, status="running"))
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code, error = 0, None
    try:
        for item in spec.get("bindings", []):
            with Path(item["path"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], item[
                    "path"
                ]
        environment = dict(os.environ, **spec.get("environment", {}))
        for index, command in enumerate(spec["commands"]):
            if stopped or any(Path(path).exists() for path in spec.get("pause_paths", [])):
                raise RuntimeError("Campaign STOP/PAUSE prevents the next command")
            write(
                directory / "status.json",
                dict(
                    receipt,
                    status="running",
                    command_index=index,
                    command=command,
                    updated_epoch=time.time(),
                ),
            )
            print(json.dumps(dict(command_index=index, argv=command)), flush=True)
            child = subprocess.Popen(command, cwd=spec["cwd"], env=environment)
            code = child.wait()
            if code:
                raise RuntimeError(f"command {index} exited {code}: {command}")
        for path in spec.get("expected_outputs", []):
            if not Path(path).is_file():
                raise RuntimeError("Missing required output: " + path)
    except BaseException as exc:
        code = code or 1
        error = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    status = "interrupted" if stopped else "failed" if code else "complete"
    complete = dict(
        receipt,
        status=status,
        exit_code=code,
        error=error,
        ended_epoch=time.time(),
        elapsed_seconds=time.time() - started,
    )
    # Completion is published before the exit marker used by the dispatcher.
    write(directory / "completion.json", complete)
    write(directory / "status.json", complete)
    (directory / "exit-code").write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
