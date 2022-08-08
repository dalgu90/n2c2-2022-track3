#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3_noteaug"
data_pseudonymize=false
#data_pseudonymize=true
train_file="train.csv"
#train_file="traindev.csv"
test_file="$1.csv"  # train, dev, or test
shuffle_train="no_shuffle_train"
#shuffle_train="shuffle_train"

# Optimization
init_step=$2

# Base BERT
#exp_name="sent_rel_Clinical-Longformer"
#bert_name="yikuan8/Clinical-Longformer"
#dataset="relation_longformer_dataset"
#batch_size=4
exp_name="sent_rel_Clinical-Longformer2"
bert_name="yikuan8/Clinical-Longformer"
dataset="relation_longformer_dataset2"
batch_size=4

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
    --test \
    --dataset=$dataset \
    --data_dir="data/${data_name}" \
    --test_file=$test_file \
    --tokenizer_name=$bert_name \
    --model="longformer_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=2048 \
    --batch_size=$batch_size \
    --num_gpus=1 \
    --init_ckpt="ckpt-${init_step}.pkl" \
    --init_step=$init_step
