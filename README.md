# n2c2-2022-track3

Repository of data preprocessing / models for n2c2 2022 Track 3: Assessment and Plan Reasoning.

# How to

## Data Preperation
- Put the dataset file `N2C2-Track3-May3.zip` and extract under `data/N2C2-Track3-May/`.
- Put the test file `n2c2_test_noLabel.csv` under `data/`.
- Put ROWID_update files `ROWID_Updated_train.csv` and `ROWID_Updated_dev.csv` under `data/ROWID_Updated/`.
- Put the MIMIC-III `NOTEEVENTS.csv` under `data/mimic3/`.

## Data Preprocessing
- To get the pseudonymized dataset, please run notebook `scripts/pseudonymize_dataset.ipynb`.
- To get the note-augmented dataset and add fake labels to the test data, please run notebook `scripts/note_augment_dataset.ipynb`.

## Training
- Run training script `train_*.sh` to train a model, and run evaluation script `eval_*.sh` to test the trained model.
    - You can change the options, including base model, data shuffle, weight sharing, and pseudonymized dataset, in the script.
