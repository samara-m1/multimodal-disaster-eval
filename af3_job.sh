#!/bin/bash
#SBATCH --job-name=af3_all
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/smansoo5/logs/af3_all_%j.out
#SBATCH --error=/scratch/smansoo5/logs/af3_all_%j.err

export HF_HOME=/scratch/smansoo5/hf_cache
export TRANSFORMERS_CACHE=/scratch/smansoo5/hf_cache
mkdir -p /scratch/smansoo5/logs

source /home/smansoo5/miniconda3/etc/profile.d/conda.sh
conda activate af3

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

/home/smansoo5/miniconda3/envs/af3/bin/python /scratch/smansoo5/run_af3_all.py

echo "Job finished: $(date)"
