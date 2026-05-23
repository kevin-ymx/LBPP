#!/bin/bash
#SBATCH --job-name=inferEb
#SBATCH --account=choiseprojec
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=10:00:00

# Kestrel: 4 nodes x 4 GPUs = 16 tasks (srun, one task per GPU)
# Each task processes CSV files where index % 16 == SLURM_PROCID

# Load env / modules (adjust for Kestrel)
# module load conda
# conda activate your_env

export MPICH_GPU_SUPPORT_ENABLED=1

INPUT_DIR="./filtered_csv_latest"
OUTPUT_DIR="./filtered_csv_latest_embeddings_finetuned"
WORLD_SIZE=16

# Set EMBEDDINGS_ONLY=--embeddings_only to skip binding energy prediction,
# or leave empty to compute both embeddings and predicted binding energy.
EMBEDDINGS_ONLY="--embeddings_only"
# EMBEDDINGS_ONLY="--embeddings_only"

srun --ntasks=${WORLD_SIZE} --cpu-bind=cores --gpus-per-task=1 bash -c '
  python inference_Eb.py \
    --filtered_csv "'"${INPUT_DIR}"'" \
    --output_dir "'"${OUTPUT_DIR}"'" \
    --rank ${SLURM_PROCID} \
    --world_size '"${WORLD_SIZE}"' \
    '"${EMBEDDINGS_ONLY}"'
'

echo "Done."
