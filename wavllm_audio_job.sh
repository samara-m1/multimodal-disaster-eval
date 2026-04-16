#!/bin/bash
#SBATCH --job-name=wavllm_audio
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/smansoo5/logs/wavllm_audio_%j.out
#SBATCH --error=/scratch/smansoo5/logs/wavllm_audio_%j.err

mkdir -p /scratch/smansoo5/logs

source /home/smansoo5/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/smansoo5/wavllm_env2   # ✅ IMPORTANT

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

cd /scratch/smansoo5/fairseq-main

bash examples/wavllm/scripts/inference_sft.sh \
    /scratch/smansoo5/wavllm_weights/final.pt \
    all_audio_320 \
    --num-workers 0   # ✅ IMPORTANT FIX

echo "Job finished: $(date)"
