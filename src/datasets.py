#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2022 n2c2 challenge track 3 dataset module

This module defines dataset classes used in 2022 n2c2 challenge track 3 and
provides the dataset retrival method.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer


_relation_labels = {'Direct': 0, 'Indirect': 1, 'Neither': 2, 'Not Relevant': 3}


def get_dataset(dataset, data_file, tokenizer_name, max_len):
    """Dataset loader method"""
    if dataset == "relation_dataset":
        return RelationDataset(data_file=data_file, tokenizer_name=tokenizer_name, max_len=max_len)
    elif dataset == "similarity_dataset":
        return SimilarityDataset(data_file=data_file, tokenizer_name=tokenizer_name, max_len=max_len)
    else:
        raise ValueError('Invalid dataset name: %s', dataset)


class BaseDataset(Dataset):
    """Base dataset for 2022 n2c2 Track 3

    This dataset module loads the data file and a transformer tokenizer. The
    dataset is not complete in that the input tokenization is not performed.
    """
    def __init__(self, data_file, tokenizer_name, max_len):
        self.data_file = data_file
        self.tokenizer_name = tokenizer_name
        # self.vocab_file = vocab_file
        self.max_len = max_len

        # Load dataset
        print(f'Load data from {data_file}... ', end='')
        self.df_data = pd.read_csv(data_file, low_memory=False)
        print(f'done {len(self.df_data)} rows')
        self.iterrows = self.df_data.iterrows()

        # Load tokenizer from transformers
        print(f'Load tokenizer {tokenizer_name}... ', end='')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print('done')

    def __len__(self):
        return len(self.df_data)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        # Get a csv row
        row = self.df_data.iloc[idx]

        # Example index and label
        example = {'row_id': row['ROW ID'],
                   'hadm_id': row['HADM ID'],
                   'label': _relation_labels[row['Relation']]}

        # Input text
        self._add_input_tokens(example, row)

        return example

    def _add_input_tokens(self, example, row):
        raise NotImplementedError()


class RelationDataset(BaseDataset):
    """A dataset for 2022 n2c2 Track 3. Sentence pair classification version"""
    def __init__(self, data_file, tokenizer_name, max_len):
        super().__init__(data_file, tokenizer_name, max_len)

    def _add_input_tokens(self, example, row):
        # Add assessment and plan in one sentence
        example['input_ids'] = self.tokenizer.encode(
            row['Assessment'], row['Plan Subsection'], add_special_tokens=True
        )[:self.max_len]

    def collate_fn(self, examples):
        max_input_len = max([len(e['input_ids']) for e in examples])
        for e in examples:
            padding = max_input_len - len(e['input_ids'])
            e['attention_mask'] = [1] * len(e['input_ids']) + [0] * padding
            e['input_ids'] += [self.tokenizer.pad_token_id] * padding

        batch = {}
        for k in examples[0].keys():
            batch[k] = torch.tensor([e[k] for e in examples])
        return batch


class SimilarityDataset(BaseDataset):
    """A dataset for 2022 n2c2 Track 3. Sentence similarity classification
    version"""
    def __init__(self, data_file, tokenizer_name, max_len):
        super().__init__(data_file, tokenizer_name, max_len)

    def _add_input_tokens(self, example, row):
        # Add assessment and plan seperately
        assessment = self.tokenizer.encode(
            row['Assessment'], add_special_tokens=True
        )[:self.max_len]
        example['input_ids_assessment'] = assessment
        plan = self.tokenizer.encode(
            row['Plan Subsection'], add_special_tokens=True
        )[:self.max_len]
        example['input_ids_plan'] = plan

    def collate_fn(self, examples):
        max_assess_len = max([len(e['input_ids_assessment']) for e in examples])
        for e in examples:
            padding = max_assess_len - len(e['input_ids_assessment'])
            e['attention_mask_assessment'] = \
                [1] * len(e['input_ids_assessment']) + [0] * padding
            e['input_ids_assessment'] += [self.tokenizer.pad_token_id] * padding
        max_plan_len = max([len(e['input_ids_plan']) for e in examples])
        for e in examples:
            padding = max_plan_len - len(e['input_ids_plan'])
            e['attention_mask_plan'] = \
                [1] * len(e['input_ids_plan']) + [0] * padding
            e['input_ids_plan'] += [self.tokenizer.pad_token_id] * padding

        batch = {}
        for k in examples[0].keys():
            batch[k] = torch.tensor([e[k] for e in examples])
        return batch


