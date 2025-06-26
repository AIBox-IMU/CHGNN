#!/bin/bash

# 定义数据集列表
datasets=("tcm_v2" "tcm_v4" "tcm_v3" "tcm_v1")

# 遍历数据集
for dataset in "${datasets[@]}"
do
    echo "Training on dataset: $dataset"
    python GNN/train.py --gpu 0 -d "$dataset" -e "$dataset" --batch_size 64 -dim 32 --dropout 0 -l 6 --num_epochs 10
done

echo "Training completed for all datasets."