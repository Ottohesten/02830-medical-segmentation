# Frederik's folder: the project on Windows

Tools that only Frederik's PC needs: Windows 11 with an NVIDIA RTX 4090, used for G3 (fine-tuning; task document
[docs/tasks/g3_finetuning.md](../docs/tasks/g3_finetuning.md)). Nothing here is needed on a Mac, on Linux or on the
DTU HPC, and nothing here changes how the rest of the project computes its results. The G3 code itself (configs,
scripts, `src/segreview/`) stays where [docs/g3_plan.md](../docs/g3_plan.md) puts it, so the group can find it and
run it on other machines too.

## Why a separate setup

`uv.lock` covers only macOS and Linux, so `uv sync` and `uv run` do not work on Windows. `windows_setup.ps1` builds a
separate environment in `.venv` with the same package versions as `uv.lock` (it checks that none differs), with
torch 2.2.2 and torchvision 0.17.2 as CUDA 12.1 builds from PyTorch's own server, plus `psutil` (to measure the
memory of nnU-Net's worker processes). `uv.lock` and `pyproject.toml` are not changed.

## Use

```powershell
powershell -ExecutionPolicy Bypass -File frederik\windows_setup.ps1    # once, and again after uv.lock changes
.venv\Scripts\python.exe -m pytest                                     # the tests
.venv\Scripts\python.exe scripts\<script>.py --config configs\<config>.yaml
```

The setup also checks that torch can use the GPU (one small 3D convolution; no training), runs the tests and
downloads the TotalSegmentator weights and the 2 smoke-test scans. Its output is in
`results/g3_smoke_gpu/setup_check.txt` (checks) and `results/g3_smoke_gpu/windows_env.txt` (package versions).
Git for Windows must be on PATH, so every run can record its git commit in `machine.json`.

## Long G3 runs

`g3_chain.ps1` runs stages of `scripts/g3_run.py` one after another and stops at the first stage that fails.
`g3_start.ps1` starts it as a process of its own, so it goes on when the terminal, VS Code or a Claude session is
closed, and `g3_stop.ps1` stops it with everything it started.

```powershell
powershell -ExecutionPolicy Bypass -File frederik\g3_start.ps1                   # start or continue (the curves)
powershell -ExecutionPolicy Bypass -File frederik\g3_start.ps1 prepare test grid # other stages
powershell -ExecutionPolicy Bypass -File frederik\g3_stop.ps1                    # stop
Get-Content data\g3_kits\logs\chain_status.txt                                   # progress per stage
Get-Content data\g3_kits\logs\chain_curves.out -Tail 5                            # the last steps
```

**Stop and continue at any time.** Stopping loses at most the training epoch in progress (about 1 min) or the scan
being evaluated: a training run saves its whole state after every epoch and continues from it, an evaluation
continues with the next scan, and finished runs are skipped. The same holds after a shutdown, a restart or a crash.
Only one chain runs at a time: `g3_start.ps1` refuses while one is running.

Close games and other heavy programs while it runs: the PC's RAM is the limit (34 GB; other programs often use
17–19 GB), and a game using the GPU's memory can make a training run fail. Stop G3 before gaming, and continue it
afterwards.

## How runs on this PC differ from the group's

- **The code is the same; the machine is not.** The model runs on a GPU instead of a CPU, and the Windows torch build
  brings cuDNN 8.8.1 (the Linux build in `uv.lock`: 8.9.2). Model outputs can therefore differ slightly from runs on
  the Macs or the HPC. Every run records its machine in `machine.json` (CONTRIBUTING.md, section 3).
- **G3 compares models only within this PC.** The original model is evaluated again here and is never compared with
  the locked kits100 numbers from the Mac.
- nnU-Net starts its data augmentation workers as new processes (spawn) instead of copies (fork) as on Linux, so a
  training run starts more slowly.
- RAM is measured as the private memory of all processes (on Linux: PSS); see `src/segreview/memory.py`.
