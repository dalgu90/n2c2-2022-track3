#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

data_name="N2C2-Track3-May3"
#test_file="train.csv"
test_file="dev.csv"

data_pseudonymize=false
#data_pseudonymize=true

exp_name="sent_sim_PubMedBERT"
bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
max_len=256
#exp_name="sent_sim_BioClinicalBERT"
#bert_name="emilyalsentzer/Bio_ClinicalBERT"
#max_len=256

if [ "$data_pseudonymize" = true ]; then
    data_name="${data_name}_pseudo"
    exp_name="${exp_name}_pseudo"
fi

#init_ckpt="9999.pkl"
#init_step=9999

python main.py \
    --test \
    --dataset="similarity_dataset" \
    --data_dir="data/${data_name}" \
    --test_file=$test_file \
    --tokenizer_name=$bert_name \
    --model="bert_sent_sim" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=$max_len \
    --batch_size=16 \
    --gpu \
    #--init_ckpt=$init_ckpt \
    #--init_step=$init_step
