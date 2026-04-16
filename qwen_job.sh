#!/bin/bash
#SBATCH --job-name=qwen_all
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/smansoo5/logs/qwen_all_%j.out
#SBATCH --error=/scratch/smansoo5/logs/qwen_all_%j.err

export HF_HOME=/scratch/smansoo5/hf_cache
export TRANSFORMERS_CACHE=/scratch/smansoo5/hf_cache
mkdir -p /scratch/smansoo5/logs

source /scratch/smansoo5/qwen_env/bin/activate

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

/scratch/smansoo5/qwen_env/bin/python /scratch/smansoo5/run_qwen_all.py

echo "Job finished: $(date)"
