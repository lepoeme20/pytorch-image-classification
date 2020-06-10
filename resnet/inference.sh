#!/bin/bash

# To training resnet
# === Set parameters ===
batch_size=512
classifier='resnet18'
class_num=10
dataset='cifar10'
data_path='/media/lepoeme20/Data/basics'
gpu_ids=(0)
image_size=32
image_channels=3
pretrained_path='/media/lepoeme20/cloud/personal/KOREA_Univ/Research/pretrained_weights/resnet'
train='false'

# Training
python inference.py --batch-size $batch_size --classifier $classifier \
    --num-classes $class_num --dataset $dataset --data-root-path $data_path \
    --device-ids ${gpu_ids[@]} --image-size $image_size \
    --save-dir $pretrained_path --train $train \
    --image-channels $image_channels
    

