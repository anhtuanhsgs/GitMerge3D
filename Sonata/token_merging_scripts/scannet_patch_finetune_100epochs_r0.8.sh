#!/bin/bash
# ScanNet full fine-tuning with token merging (r=0.80)
# Merges 80% of tokens (retains 20% of tokens)

sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
