#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3"
data_pseudonymize=false
#data_pseudonymize=true
train_file="train.csv"
#train_file="traindev.csv"
dev_file="dev.csv"
shuffle_train="no_shuffle_train"
#shuffle_train="shuffle_train"

# Model
weight_sharing=true
#weight_sharing=false

# Optimization
num_gpus=1
#num_gpus=2

# Base BERT
model="bert_sent_sim"
if [[ $1 == "PubMedBERT" ]]; then
    exp_name="sent_sim_PubMedBERT"
    bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    batch_size=32
elif [[ $1 == "BioClinicalBERT" ]]; then
    exp_name="sent_sim_BioClinicalBERT"
    bert_name="emilyalsentzer/Bio_ClinicalBERT"
    batch_size=32
elif [[ $1 == "bert-base-uncased" ]]; then
    exp_name="sent_sim_bert-base-uncased"
    bert_name="bert-base-uncased"
    batch_size=32
fi

# Process options
if [ "$weight_sharing" = false ]; then
    model="${model/_sim/_sim2}"
    exp_name="${exp_name/_sim/_sim2}"
fi
if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi
batch_size=$((2 * batch_size))
if [[ $num_gpus == 2 ]]; then
    exp_name="${exp_name}_2gpu"
fi
if [[ $shuffle_train == "no_shuffle_train" ]]; then
    exp_name="${exp_name}_noshuffle"
fi
if [[ $train_file == "traindev.csv" ]]; then
    exp_name="${exp_name}_traindev"
fi

# Run!
python main.py \
    --dataset="similarity_dataset" \
    --data_dir="data/${data_name}" \
    --train_file=$train_file \
    --dev_file=$dev_file \
    --tokenizer_name=$bert_name \
    --model=$model \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=256 \
    --train_bert \
    --batch_size=$batch_size \
    --${shuffle_train} \
    --training_step=10000 \
    --lr=0.0001 \
    --warmup_updates=1000 \
    --display_iter=25 \
    --eval_iter=1000 \
    --ckpt_max_to_keep=10 \
    --num_gpus=$num_gpus \
