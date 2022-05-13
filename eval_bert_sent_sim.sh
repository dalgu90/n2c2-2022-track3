#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3"
#test_file="train.csv"
test_file="dev.csv"
data_pseudonymize=false
#data_pseudonymize=true

# Model
model="bert_sent_sim"
#weight_sharing=true
weight_sharing=false

# Base BERT
exp_name="sent_sim_PubMedBERT"
bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#exp_name="sent_sim_BioClinicalBERT"
#bert_name="emilyalsentzer/Bio_ClinicalBERT"

if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi

if [ "$weight_sharing" = false ]; then
    model="${model/_sim/_sim2}"
    exp_name="${exp_name/_sim/_sim2}"
fi

#init_step=9999
#init_ckpt="ckpt-${init_step}.pkl"

python main.py \
    --test \
    --dataset="similarity_dataset" \
    --data_dir="data/${data_name}" \
    --test_file=$test_file \
    --tokenizer_name=$bert_name \
    --model=$model \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=256 \
    --batch_size=16 \
    --gpu \
    #--init_ckpt=$init_ckpt \
    #--init_step=$init_step
