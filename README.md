# n2c2-2022-track3

Repository of data preprocessing / models for n2c2 2022 Track 3: Assessment and Plan Reasoning.

# How to

## Data Preperation
- Put the dataset file `N2C2-Track3-May3.zip` and extract under `data/N2C2-Track3-May/`.
- Put the test file `n2c2_test_noLabel.csv` under `data/`.
- Put ROWID_update files `ROWID_Updated_train.csv` and `ROWID_Updated_dev.csv` under `data/ROWID_Updated/`.
- Put the MIMIC-III `NOTEEVENTS.csv` under `data/mimic3/`.

## Data Preprocessing
- To get the pseudonymized dataset, run notebook `scripts/pseudonymize_dataset.ipynb`.
- To get the note-augmented dataset and add fake labels to the test data, run notebook `scripts/note_augment_dataset.ipynb`.

## Training
- Run training script `train_*_sent_rel.sh` to train models.
    - When running `train_bert_sent_rel.sh`, please pass the base BERT model.
    - You can change the options, including base model, data shuffle, weight sharing, and pseudonymized dataset, in the script.
- To test the trained models, run evaluation script `eval_*_sent_rel.sh`.
    - Please pass the model, the split to evaluate, and the ckpt iteration.
    - For postprocessing, get predictions on train/dev(/test) set.

## Postprocessing
- For basic analysis on the train/dev predictions, please run `script/results_basic_analysis.ipynb`.
- For ensemble method, please run `script/results_ensemble.ipynb`.
- For Bayesian inference method, please run notebook `scripts/results_bayesian_inference.ipynb`.
