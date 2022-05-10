#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

exp_name="sent_rel_PubMedBERT"
bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
max_len=384
#exp_name="sent_rel_BioClinicalBERT"
#bert_name="emilyalsentzer/Bio_ClinicalBERT"
#max_len=384

#init_ckpt="9999.pkl"
#init_step=9999

python main.py \
    --test \
    --dataset="relation_dataset" \
    --data_dir="data/N2C2-Track3-May3" \
    --test_file="dev.csv" \
    --tokenizer_name=$bert_name \
    --model="bert_sent_rel" \
    --bert_name=$bert_name \
    --output_dir="results/${exp_name}" \
    --max_len=$max_len \
    --batch_size=16 \
    --gpu \
    #--init_ckpt=$init_ckpt \
    #--init_step=$init_step
