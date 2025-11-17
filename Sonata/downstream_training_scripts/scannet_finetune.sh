#!/bin/bash
export WANDB_API_KEY=""
export config=$1
echo $config
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /netscratch/ttran/projects/3DTokenMerging/envs/pointcept
sh scripts/train.sh -g 1  -d sonata \
        -c $config -n $config \
        -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth

