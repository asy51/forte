#!/bin/bash
#SBATCH --account=vxc204_aisc
#SBATCH --partition=aisc

###SBATCH --account=llm_workshop2024
###SBATCH --reservation=llm24
###SBATCH --partition=aisc_short

#SBATCH --job-name=forte_oai
#SBATCH --time=1-00:00:00
###SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1

source /home/asy51/.bashrc
conda activate hf
cd /home/asy51/repos/forte

echo "$(hostname): $(which python)"

srun python -u main.py --id_images_directories './symlinks/oai_dess_left' \
    --id_images_names oai_dess_left \
    --ood_images_directories './symlinks/oai_dess_right' \
    --ood_images_names oai_dess_right \
    --batch_size 2 \
    --device cuda:0 \
    --num_seeds 1 \

srun python -u main.py --id_images_directories './symlinks/oai_tse_left' \
    --id_images_names oai_tse_left \
    --ood_images_directories './symlinks/oai_tse_right' \
    --ood_images_names oai_tse_right \
    --batch_size 10 \
    --device cuda:0 \
    --num_seeds 1 \

# srun python -u main.py --id_images_directories './symlinks/test' \
#     --id_images_names test \
#     --ood_images_directories './symlinks/test' \
#     --ood_images_names test \
#     --batch_size 10 \
#     --device cuda:0 \
#     --num_seeds 1 \