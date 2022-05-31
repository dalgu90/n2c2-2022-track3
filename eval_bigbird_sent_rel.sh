#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3"
test_file="train.csv"
test_file="dev.csv"
data_pseudonymize=false
#data_pseudonymize=true

# Base BERT
#exp_name="sent_rel_Clinical-Longformer"
#bert_name="yikuan8/Clinical-Longformer"
#dataset="relation_longformer_dataset"
#batch_size=32
exp_name="sent_rel_Clinical-Longformer2_8"
bert_name="yikuan8/Clinical-Longformer"
dataset="relation_longformer_dataset2"
batch_size=16

if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi

#init_step=2000
#init_ckpt="ckpt-${init_step}.pkl"

python main.py \
    --test \
    --dataset=$dataset \
    --data_dir="data/${data_name}" \
    --test_file=$test_file \
    --tokenizer_name=$bert_name \
    --model="longformer_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=384 \
    --batch_size=16 \
    --gpu \
    #--init_ckpt=$init_ckpt \
    #--init_step=$init_step
