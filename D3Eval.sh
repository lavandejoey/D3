#!/bin/bash
#SBATCH --job-name=D3Eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A40,L40S
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_DeFake_ViTB32"
mkdir -p "${result_dir}"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMockBin"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
data_entry_csv="/projects/hi-paris/DeepFakeDataset/frames_index.csv"
done_csv_list=("results")

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310

srun python3 -Wignore D3Eval.py \
--data_root "${data_root}" \
--results "${result_dir}" \
--ckpt "ckpt/classifier.pth" \
--data_csv ${data_entry_csv} \
--done_csv_list "${done_csv_list[@]}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"