#!/bin/bash
#SBATCH --job-name=hrl_train
#SBATCH --output=logs/hrl_train_%j.log
#SBATCH --error=logs/hrl_train_%j.err
#SBATCH --gres=gpu:a100:1       # Request one A100 GPU
#SBATCH --cpus-per-task=8       # Adjust as needed
#SBATCH --mem=32G               # Memory allocation
#SBATCH --time=48:00:00         # Max run time
#SBATCH --partition=gpu         # Use your uni's GPU partition name

# Load environment modules
module load anaconda
source ~/hrl_env/bin/activate  

# Log in to Weights & Biases (optional)
export WANDB_API_KEY=
export WANDB_MODE=

# Optional: record metadata
echo "Job started on $(hostname) at $(date)"

# Run training script
python train_hrl_agent.py --config configs/ppo_hrl.yaml

# Optional: record completion time
echo "Job finished at $(date)"
