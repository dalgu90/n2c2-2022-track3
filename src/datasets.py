#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2022 n2c2 challenge track 3 dataset module

This module defines dataset classes used in 2022 n2c2 challenge track 3 and
provides the dataset retrival method.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer


_relation_labels = {'Direct': 0, 'Indirect': 1, 'Neither': 2, 'Not Relevant': 3}
_relation_labels_ms_relevant = {'Relevant': 0, 'Not Relevant': 1}
_relation_labels_ms_mentioned = {'Mentioned': 0, 'Neither': 1}
_relation_labels_ms_direct = {'Direct': 0, 'Indirect': 1}
_relation_labels_ms_threeprob = {'Direct': 0, 'Indirect': 1, 'Neither': 2}


def get_dataset(args, split="train"):
    """Dataset loader method"""
    if split == "train":
        data_file = os.path.join(args.data_dir, args.train_file)
    elif split == "dev":
        data_file = os.path.join(args.data_dir, args.dev_file)
    elif split == "test":
        data_file = os.path.join(args.data_dir, args.test_file)
    else:
        raise ValueError(f"Invalid split: {split}")

    if args.dataset == "relation_dataset":
        return RelationDataset(data_file=data_file,
                               tokenizer_name=args.tokenizer_name,
                               max_len=args.max_len)
    elif args.dataset == "similarity_dataset":
        return SimilarityDataset(data_file=data_file,
                                 tokenizer_name=args.tokenizer_name,
                                 max_len=args.max_len)
    elif args.dataset == "relation_longformer_dataset":
        return RelationLongformerDataset(data_file=data_file,
                                         tokenizer_name=args.tokenizer_name,
                                         max_len=args.max_len)
    elif args.dataset == "relation_longformer_dataset2":
        return RelationLongformerDataset2(data_file=data_file,
                                          tokenizer_name=args.tokenizer_name,
                                          max_len=args.max_len)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")


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
        self.relation_labels = _relation_labels

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
                   'label': self.relation_labels[row['Relation']]}

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
        return_dict = self.tokenizer(
            row['Assessment'], row['Plan Subsection'], add_special_tokens=True
        )
        example['input_ids'] = return_dict['input_ids'][:self.max_len]
        example['token_type_ids'] = return_dict['token_type_ids'][:self.max_len]
        example['attention_mask'] = return_dict['attention_mask'][:self.max_len]

    def collate_fn(self, examples):
        max_input_len = max([len(e['input_ids']) for e in examples])
        for e in examples:
            padding = max_input_len - len(e['input_ids'])
            e['attention_mask'] = [1] * len(e['input_ids']) + [0] * padding
            e['input_ids'] += [self.tokenizer.pad_token_id] * padding
            e['token_type_ids'] += [self.tokenizer.pad_token_id] * padding

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


class RelationLongformerDataset(BaseDataset):
    """A dataset for 2022 n2c2 Track 3. Sentence pair classification version"""
    def __init__(self, data_file, tokenizer_name, max_len):
    # def __init__(self, data_file, tokenizer_name, max_len, attn_config_file):
        super().__init__(data_file, tokenizer_name, max_len)
        self.special_token_ids = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id
        ]
        # with open(attn_config_file, 'r') as fd:
            # self.attn_config = json.load(fd)

        self.input_col_names = ['Assessment', 'Plan Subsection', 'present_history', 'chief_complaint', 'hospital_course']
        for col in self.input_col_names:
            self.df_data[col] = self.df_data[col].map(
                lambda x: "empty" if (isinstance(x, float) and np.isnan(x)) else self._preprocess(x))

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(' +', ' ', text)
        return text

    def _plan_problem_idx(self, text):
        problem_delimiters = [':', '. ', '- ']
        lines = text.split('\n')
        for w in problem_delimiters:
            if lines[0].find(w) != -1:
                return lines[0].index(w)
        if len(lines) == 1:
            return len(text)
        for w in problem_delimiters:
            if lines[1].find(w) != -1:
                return text.index(w)
        return len(lines[0])

    def _add_input_tokens(self, example, row):
        # Token id arrangement
        #                       Assess            Plan         Others
        # Input IDs  : 0(cls) | tokens | 2(sep) | tokens | 2 | tokens | 2 | tokens | 2 | tokens | 2
        # Token type : 0      | 0 * n  | 0      | 1 * n  | 0 | 2 * n  | 0 | 3 * n  | 0 | 4 * n  | 0
        # Global att : 1      | 1 * n  | 1      | 1 / 0  | 0 | 0 * n  | 0 | 0 * n  | 0 | 0 * n  | 0
        example['input_ids'] = [self.tokenizer.cls_token_id]
        example['token_type_ids'] = [0]
        example['global_attention_mask'] = [1]
        for i, col in enumerate(self.input_col_names):
            text = row[col]
            if i == 0:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                example['global_attention_mask'] += [1] * (len(tokens) + 1)
            elif i == 1:
                prob_idx = self._plan_problem_idx(text)
                text1, text2 = text[:prob_idx], text[prob_idx:]
                tokens1 = self.tokenizer.encode(text1, add_special_tokens=False)
                tokens2 = self.tokenizer.encode(text2, add_special_tokens=False)
                tokens = tokens1 + tokens2
                example['global_attention_mask'] += [1] * len(tokens1) + [0] * (len(tokens2) + 1)
            else:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                example['global_attention_mask'] += [0] * (len(tokens) + 1)

            example['input_ids'] += tokens
            example['input_ids'].append(self.tokenizer.sep_token_id)
            example['token_type_ids'] += [i] * len(tokens)
            example['token_type_ids'].append(0)

        # Truncate to max_len
        example['input_ids'] = example['input_ids'][:self.max_len]
        example['token_type_ids'] = example['token_type_ids'][:self.max_len]
        example['global_attention_mask'] = example['global_attention_mask'][:self.max_len]

    def collate_fn(self, examples):
        max_input_len = max([len(e['input_ids']) for e in examples])
        for e in examples:
            padding = max_input_len - len(e['input_ids'])
            e['attention_mask'] = [1] * len(e['input_ids']) + [0] * padding
            e['input_ids'] += [self.tokenizer.pad_token_id] * padding
            e['token_type_ids'] += [0] * padding
            e['global_attention_mask'] += [0] * padding

        batch = {}
        for k in examples[0].keys():
            batch[k] = torch.tensor([e[k] for e in examples])
        return batch


class RelationLongformerDataset2(BaseDataset):
    """A dataset for 2022 n2c2 Track 3. Sentence pair classification version"""
    def __init__(self, data_file, tokenizer_name, max_len):
    # def __init__(self, data_file, tokenizer_name, max_len, attn_config_file):
        super().__init__(data_file, tokenizer_name, max_len)
        self.special_token_ids = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id
        ]
        # with open(attn_config_file, 'r') as fd:
            # self.attn_config = json.load(fd)

        self.input_col_names = ['Plan Subsection', 'Assessment', 'chief_complaint', 'hospital_course', 'present_history']
        for col in self.input_col_names:
            self.df_data[col] = self.df_data[col].map(
                lambda x: "empty" if (isinstance(x, float) and np.isnan(x)) else self._preprocess(x))

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(' +', ' ', text)
        return text

    def _plan_problem_idx(self, text):
        problem_delimiters = [':', '. ', '- ']
        lines = text.split('\n')
        for w in problem_delimiters:
            if lines[0].find(w) != -1:
                return lines[0].index(w)
        if len(lines) == 1:
            return len(text)
        for w in problem_delimiters:
            if lines[1].find(w) != -1:
                return text.index(w)
        return len(lines[0])

    def _add_input_tokens(self, example, row):
        # Token id arrangement
        #                       Plan         Assess            Others
        # Input IDs  : 0(cls) | tokens | 2 | tokens | 2(sep) | tokens | 2 | tokens | 2 | tokens | 2
        # Token type : 0      | 0 * n  | 0 | 1 * n  | 0      | 2 * n  | 0 | 2 * n  | 0 | 2 * n  | 0
        # Global att : 1      | 1 * n  | 1 | 1 * n  | 1      | 0 * n  | 0 | 0 * n  | 0 | 0 * n  | 0
        example['input_ids'] = [self.tokenizer.cls_token_id]
        example['token_type_ids'] = [0]
        example['global_attention_mask'] = [1]
        for i, col in enumerate(self.input_col_names):
            text = row[col]
            if i == 0:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[:250]
                example['global_attention_mask'] += [1] * (len(tokens) + 1)
            elif i == 1:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[:150]
                example['global_attention_mask'] += [1] * (len(tokens) + 1)
            elif i == 2:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                example['global_attention_mask'] += [0] * (len(tokens) + 1)
            elif i == 3:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[:1240]
                example['global_attention_mask'] += [0] * (len(tokens) + 1)
            elif i == 4:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[:590]
                example['global_attention_mask'] += [0] * (len(tokens) + 1)
            else:
                raise ValueError()

            example['input_ids'] += tokens
            example['input_ids'].append(self.tokenizer.sep_token_id)
            example['token_type_ids'] += [i if i < 2 else 2] * len(tokens)
            example['token_type_ids'].append(0)

        # Truncate to max_len
        example['input_ids'] = example['input_ids'][:self.max_len]
        example['token_type_ids'] = example['token_type_ids'][:self.max_len]
        example['global_attention_mask'] = example['global_attention_mask'][:self.max_len]

    def collate_fn(self, examples):
        max_input_len = max([len(e['input_ids']) for e in examples])
        for e in examples:
            padding = max_input_len - len(e['input_ids'])
            e['attention_mask'] = [1] * len(e['input_ids']) + [0] * padding
            e['input_ids'] += [self.tokenizer.pad_token_id] * padding
            e['token_type_ids'] += [0] * padding
            e['global_attention_mask'] += [0] * padding

        batch = {}
        for k in examples[0].keys():
            batch[k] = torch.tensor([e[k] for e in examples])
        return batch
