#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3"
data_pseudonymize=false
#data_pseudonymize=true

# Base BERT
#exp_name="sent_rel_PubMedBERT"
#bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#batch_size=32
#exp_name="sent_rel_BioClinicalBERT"
#bert_name="emilyalsentzer/Bio_ClinicalBERT"
#batch_size=32
#exp_name="sent_rel_BlueBERT-Base"
#bert_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
#batch_size=32
exp_name="sent_rel_BlueBERT-Large"
bert_name="bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
batch_size=16
#exp_name="sent_rel_bert-base-uncased"
#bert_name="bert-base-uncased"
#batch_size=32

if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi

python main.py \
    --dataset="relation_dataset" \
    --data_dir="data/${data_name}" \
    --train_file="train.csv" \
    --dev_file="dev.csv" \
    --tokenizer_name=$bert_name \
    --model="bert_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=384 \
    --train_bert \
    --batch_size=$batch_size \
    --training_step=10000 \
    --lr=0.0001 \
    --warmup_updates=1000 \
    --display_iter=25 \
    --eval_iter=1000 \
    --gpu
