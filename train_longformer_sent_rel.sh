#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3_noteaug"
data_pseudonymize=false
#data_pseudonymize=true
train_file="train.csv"
#train_file="traindev.csv"
dev_file="dev.csv"

# Optimization
shuffle_train="no_shuffle_train"
#shuffle_train="shuffle_train"

# Base BERT
#exp_name="sent_rel_Clinical-Longformer"
#bert_name="yikuan8/Clinical-Longformer"
#dataset="relation_longformer_dataset"
#batch_size=16
exp_name="sent_rel_Clinical-Longformer2"
bert_name="yikuan8/Clinical-Longformer"
dataset="relation_longformer_dataset2"
batch_size=16

# Process options
if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi
if [[ $shuffle_train == "no_shuffle_train" ]]; then
    exp_name="${exp_name}_noshuffle"
fi
if [[ $train_file == "traindev.csv" ]]; then
    exp_name="${exp_name}_traindev"
fi

# Run!
python main.py \
    --dataset=$dataset \
    --data_dir="data/${data_name}" \
    --train_file=$train_file \
    --dev_file=$dev_file \
    --tokenizer_name=$bert_name \
    --model="longformer_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=2048 \
    --train_bert \
    --batch_size=$batch_size \
    --${shuffle_train} \
    --training_step=10000 \
    --lr=0.0001 \
    --warmup_updates=1000 \
    --display_iter=25 \
    --eval_iter=1000 \
    --ckpt_max_to_keep=10 \
    --num_gpus=2
