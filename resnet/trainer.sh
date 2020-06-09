#!/bin/bash

# To training resnet
# === Set parameters ===
batch_size=256
classifier='resnet18'
class_num=10
dataset='cifar10'
data_path='/media/lepoeme20/Data/basics'
epochs=150
gpu_ids=(0 1)
image_size=32
image_channels=3
pretrained_path='./pretrained_models/'
train='true'

# Training
python main.py --batch-size $batch_size --classifier $classifier \
    --num-classes $class_num --dataset $dataset --data-root-path $data_path \
    --epochs $epochs --device-ids ${gpu_ids[@]} --image-size $image_size \
    --pretrained-dir $pretrained_path --train $train \
    --image-channels $image_channels
    
