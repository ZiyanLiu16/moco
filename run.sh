#!/bin/bash
#PBS -N moco-pretrain
#PBS -q debug
#PBS -A <your_project_name>
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -V

# --- user-configurable paths -------------------------------------------------
REPO_ROOT="/lus/eagle/projects/PBML/ziyan/moco"
DATASET_DIR="/lus/eagle/projects/PBML/ziyan/dataset"
JOB_NAME="${JOB_NAME:-moco_rvlcdip}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs}"
RUN_DIR="${OUT_ROOT}/${JOB_NAME}"

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"

# --- environment setup -------------------------------------------------------
# If your modules/venv setup is initialized in ~/.bashrc, bring them in.
if [ -f "${HOME}/.bashrc" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
fi


# Ensure conda is available in batch jobs
if ! command -v conda >/dev/null 2>&1; then
  module load conda
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate the fixed Conda environment path unless overridden.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/lus/eagle/projects/PBML/ziyan/envs/moco}"
conda activate "${CONDA_ENV_PATH}"

set -euo pipefail


# Make sure this checkout is importable.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Networking setup for torch.distributed on a single Aurora node.
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT="${MASTER_PORT:-23456}"
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE=1
export RANK=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"

# --- training command --------------------------------------------------------
python -u main_moco.py "${DATASET_DIR}" \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
  --workers 1 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  2>&1 | tee "${RUN_DIR}/train.log"