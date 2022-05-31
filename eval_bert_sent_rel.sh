#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# Dataset
data_name="N2C2-Track3-May3"
#test_file="train.csv"
test_file="dev.csv"
data_pseudonymize=false
#data_pseudonymize=true

# Base BERT
#exp_name="sent_rel_PubMedBERT"
#bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#exp_name="sent_rel_BioClinicalBERT"
#bert_name="emilyalsentzer/Bio_ClinicalBERT"
#exp_name="sent_rel_BlueBERT-Base"
#bert_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
exp_name="sent_rel_BlueBERT-Large"
bert_name="bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
#exp_name="sent_rel_bert-base-uncased"
#bert_name="bert-base-uncased"

if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi

init_step=7000
init_ckpt="ckpt-${init_step}.pkl"

python main.py \
    --test \
    --dataset="relation_dataset" \
    --data_dir="data/${data_name}" \
    --test_file=$test_file \
    --tokenizer_name=$bert_name \
    --model="bert_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=384 \
    --batch_size=16 \
    --gpu \
    --init_ckpt=$init_ckpt \
    --init_step=$init_step
