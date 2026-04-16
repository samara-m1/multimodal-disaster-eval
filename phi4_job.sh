#!/bin/bash
#SBATCH --job-name=phi4_all
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/smansoo5/logs/phi4_all_%j.out
#SBATCH --error=/scratch/smansoo5/logs/phi4_all_%j.err

export HF_HOME=/scratch/smansoo5/hf_cache
export TRANSFORMERS_CACHE=/scratch/smansoo5/hf_cache
mkdir -p /scratch/smansoo5/logs

source /home/smansoo5/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/smansoo5/phi4_env

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

/scratch/smansoo5/phi4_env/bin/python /scratch/smansoo5/run_phi4_all.py

echo "Job finished: $(date)"
