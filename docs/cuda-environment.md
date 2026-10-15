# Reproducible CUDA node environment

Use Linux x86-64 with Python 3.11. The CPU reference environment remains in
`uv.lock`. The two hash-locked CUDA profiles retain PyTorch 2.2.2, TorchVision
0.17.2 and the common dependencies from that lock; they add the matching NVIDIA
runtime libraries. These profiles were resolved and dry-run checked for Linux;
actual GPU installation and execution must be verified on the node.

Choose one profile after recording `nvidia-smi`:

| Profile | Binaries | Linux driver floor for CUDA minor-version compatibility |
| --- | --- | --- |
| `cu121` | PyTorch 2.2.2+cu121, TorchVision 0.17.2+cu121 | 525.60.13 |
| `cu118` | PyTorch 2.2.2+cu118, TorchVision 0.17.2+cu118 | 450.80.02 |

These are compatibility floors, not proof that the node works. Follow the
[NVIDIA 12.1](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)
or [11.8 compatibility guidance](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/)
and run the acceptance checks below. Both binary families are listed in
[PyTorch's 2.2.2 instructions](https://pytorch.org/get-started/previous-versions/#v222).

From a clean checkout, with uv 0.13.1:

```sh
KGR4VRD_TORCH_BACKEND=cu121  # choose cu118 if appropriate for the measured driver
uv venv .venv-cuda --python 3.11
uv pip sync --python .venv-cuda/bin/python --require-hashes \
  --torch-backend "$KGR4VRD_TORCH_BACKEND" \
  "requirements/linux-$KGR4VRD_TORCH_BACKEND.txt"
uv build --wheel
uv pip install --python .venv-cuda/bin/python --no-deps dist/kgr4vrd-0.1.0-py3-none-any.whl
uv pip check --python .venv-cuda/bin/python
export PATH="$PWD/.venv-cuda/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The `--torch-backend` argument selects the official PyTorch index for torch and
torchvision; retain it when syncing these files. Other packages use PyPI. A later
sync removes the installed project wheel, so install the wheel again afterward.
Use this environment's `python` and `kgr4vrd` directly throughout the campaign.
`uv run --locked` and `uv sync --locked` belong to the separate CPU environment.

Record the exact revision and installed environment, then run actual kernels:

```sh
mkdir -p artifacts/node-acceptance
git rev-parse HEAD > artifacts/node-acceptance/source-commit.txt
nvidia-smi > artifacts/node-acceptance/nvidia-smi.txt
python -m torch.utils.collect_env > artifacts/node-acceptance/torch-environment.txt
uv pip freeze --python .venv-cuda/bin/python > artifacts/node-acceptance/installed.txt
python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(1, device="cuda"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x)'
KGR4VRD_CUDA_TEST=1 python -m pytest -q tests/test_cuda_integration.py tests/test_vision.py -k 'cuda or registered_heads' --junitxml=artifacts/node-acceptance/cuda.xml
python -m pytest -q
```

The opt-in tests exercise FP32 forward/backward for both registered heads, visual
device routing with a small frozen test encoder, and exact resumed-versus-complete
training on either side of epoch 3. They use the production numerical defaults and
the exported cuBLAS workspace, record numerical flags in JUnit properties, and
restore thread/RNG state. Exact resume is a node acceptance test, not a promise of
cross-platform bitwise equality. Requesting CUDA when unavailable fails. The fake
encoder check does not replace inference with the acquired ResNet-50 V2 weights.
Set all three integration variables described in the [node handoff](compute-node-handoff.md)
and require no skipped integration tests before accepting the environment.

Complete `configs/nesy4vrd-cuda.toml` with the acquired input paths, and use
`configs/campaign-cuda.toml` for planning. The CUDA base inherits the same scientific
defaults as the CPU base; only the device differs. Set it before any development
run. Do not switch hardware, CUDA profile or numerical settings within a frozen
campaign without revisiting the affected development evidence.

## Regenerating the locks

Regeneration is a reviewed dependency change. With uv 0.13.1, preserve the existing
CPU lock's common versions and resolve only the CUDA additions:

```sh
uv export --locked --all-extras --group dev --no-emit-project \
  --no-emit-package torch --no-emit-package torchvision --no-hashes --no-header \
  --output-file requirements/base-constraints.txt
uv pip compile pyproject.toml --all-extras --group dev --no-sources \
  --torch-backend cu118 --constraints requirements/base-constraints.txt \
  --python-platform x86_64-manylinux_2_17 --python-version 3.11 \
  --generate-hashes --emit-index-url --emit-index-annotation \
  --output-file requirements/linux-cu118.txt
uv pip compile pyproject.toml --all-extras --group dev --no-sources \
  --torch-backend cu121 --constraints requirements/cuda121-constraints.txt \
  --python-platform x86_64-manylinux_2_17 --python-version 3.11 \
  --generate-hashes --emit-index-url --emit-index-annotation \
  --output-file requirements/linux-cu121.txt
```

The CUDA 12.1 constraints also pin its otherwise floating nvJitLink dependency to
12.1.105. Each target can be checked without altering an environment by passing
`--dry-run --python-platform x86_64-manylinux_2_17` to the `uv pip sync` command.
