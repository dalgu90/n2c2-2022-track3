# n2c2-2022-track3

Repository of data preprocessing / models for n2c2 2022 Track 3: Assessment and Plan Reasoning.

### How to

- Put the dataset file `N2C2-Track3-May3.zip` and extract under `data/N2C2-Track3-May/`.
- To get the pseudonymized dataset, please run notebook `scripts/pseudonymize_dataset.ipynb`.
- Run training script `train_*.sh` to train a model, and run evaluation script `eval_*.sh` to test the trained model.
    - You can change the options of base model, weight sharing, pseudonymized dataset in the script.
