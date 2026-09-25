#!/bin/bash
# LSF job for the DTU HPC cluster: the whole G1 pipeline of a config (scripts/run_g1.py): inference (skipped where
# already done), TTA, the second model, Dice, uncertainty scores and heatmaps, evaluation and figures.
#
# NOT TESTED ON THE CLUSTER YET. The queue names, resources and module setup follow the DTU HPC documentation,
# but the job has never been submitted. Check the first run closely (see README, "Running on the DTU HPC").
#
# Submit from the repository root (after the one-time setup in the README):
#   mkdir -p logs && bsub < jobs/hpc_g1.sh
# Another config:  CONFIG=configs/kits100.yaml bsub < jobs/hpc_g1.sh   (bsub passes the variable on)
# Tip: run jobs/hpc_inference.sh first; this job then only does the remaining steps.

#BSUB -J segreview_g1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/g1_%J.out
#BSUB -e logs/g1_%J.err

set -euo pipefail
CONFIG="${CONFIG:-configs/hpc.yaml}"
cd "${LS_SUBCWD:-.}"                            # LSF: the folder bsub was called from (the repository root)
export PATH="$HOME/.local/bin:$PATH"            # where the uv installer puts uv
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}" # use only the cores LSF gave the job (-n)

echo "Job ${LSB_JOBID:-local} on $(hostname), config $CONFIG, $(date)"
nvidia-smi || echo "WARNING: no GPU visible"
uv run --frozen python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
uv run --frozen python scripts/run_g1.py --config "$CONFIG"
echo "Done $(date)"
