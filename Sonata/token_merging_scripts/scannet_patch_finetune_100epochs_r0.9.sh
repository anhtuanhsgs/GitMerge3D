#!/bin/bash
# ScanNet full fine-tuning with token merging (r=0.90)
# Merges 90% of tokens (retains 10% of tokens)

sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
