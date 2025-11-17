#!/bin/bash
# S3DIS linear probing with token merging (r=0.80, weighted patch)
# Merges 80% of tokens (retains 20% of tokens)

sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-r80-s3dis-lin \
    -n gitmerge3d-patch-r80-s3dis-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh