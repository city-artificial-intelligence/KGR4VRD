import os
import subprocess
import sys

from kgr4vrd.cli import main


def test_import_has_no_heavy_dependencies_or_writes(tmp_path):
    code = (
        "import pathlib, sys; import kgr4vrd; "
        "assert not set(('torch', 'torchvision', 'matplotlib')).intersection(sys.modules); "
        "assert not list(pathlib.Path.cwd().iterdir())"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def test_actionable_cli_exit_codes(tmp_path):
    import pytest

    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(tmp_path / "absent.toml")])
    assert error.value.code == 3
    invalid = tmp_path / "invalid.toml"
    invalid.write_text('schema_version="unsupported/9"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(invalid)])
    assert error.value.code == 2
