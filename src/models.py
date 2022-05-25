#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2022 n2c2 challenge track 3 models

This module defines models used in 2022 n2c2 challenge track 3 and provides the
model retrieval method.
"""

import torch
from torch import nn
import transformers
from transformers import AutoModel


def get_model(model, bert_name, num_cls_layers):
    if model == "bert_sent_rel":
        return BertSentenceRelation(bert_name, num_cls_layers)
    elif model == "bert_sent_sim":
        return BertSentenceSimilarity(bert_name, num_cls_layers)
    elif model == "bert_sent_sim2":
        return BertSentenceSimilarity2(bert_name, num_cls_layers)
    else:
        raise ValueError('Invalid model name: %s', model)


class BertBase(nn.Module):
    """Base model class for 2022 n2c2 Track 3.

    This module load a transformer BERT model and defines some methods that are
    shared among other models.
    """
    def __init__(self, bert_name):
        super().__init__()
        self.bert_name = bert_name

        # Load the BERT module
        print(f'Load the BERT module {bert_name}... ', end='')
        self.bert = AutoModel.from_pretrained(bert_name)
        print('done')

    def get_param_group(self, train_bert, finetune_factor):
        if not train_bert:
            param_group = [{'params': list(self.parameters()), 'lr_factor': 1.0}]
        else:
            bert_params = list(self.bert.parameters())
            bert_params_set = set(bert_params)
            other_params = [p for p in list(self.parameters()) if p not in bert_params_set]
            param_group = [{'params': other_params, 'lr_factor': 1.0},
                           {'params': bert_params, 'lr_factor': finetune_factor}]
        return param_group


class BertSentenceRelation(BertBase):
    """A sentence pair classification model for 2022 n2c2 Track 3"""
    def __init__(self, bert_name, num_cls_layers):
        super().__init__(bert_name)

        # Define the classification header
        self.cls_dropout = nn.Dropout(p=0.1)
        cls_layers = []
        for i in range(num_cls_layers):
            if i != num_cls_layers - 1:
                in_features = out_features = self.bert.config.hidden_size
            else:
                in_features, out_features = self.bert.config.hidden_size, 4
            cls_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            if i != num_cls_layers - 1:
                cls_layers.append(nn.ReLU())
        self.classifier = nn.ModuleList(cls_layers)

    def forward(self, batch):
        pooler_output = self.bert.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )['pooler_output']
        features = self.cls_dropout(pooler_output)
        for l in self.classifier:
            features = l(features)
        return features


class BertSentenceSimilarity(BertBase):
    """A sentence similarity classification model for 2022 n2c2 Track 3

    This model is the weight sharing version where the asseseement and plan are
    fed into the same BERT model."""
    def __init__(self, bert_name, num_cls_layers):
        super().__init__(bert_name)

        # Define the classification header
        self.cls_dropout = nn.Dropout(p=0.1)
        cls_layers = []
        for i in range(num_cls_layers):
            if i == 0:
                in_features = 2 * self.bert.config.hidden_size
                out_features = self.bert.config.hidden_size
            elif i != num_cls_layers - 1:
                in_features = out_features = self.bert.config.hidden_size
            else:
                in_features, out_features = self.bert.config.hidden_size, 4
            cls_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            if i != num_cls_layers - 1:
                cls_layers.append(nn.ReLU())
        self.classifier = nn.ModuleList(cls_layers)

    def forward(self, batch):
        assess_pooler_output = self.bert(
            input_ids=batch['input_ids_assessment'],
            attention_mask=batch['attention_mask_assessment']
        )['pooler_output']
        plan_pooler_output = self.bert(
            input_ids=batch['input_ids_plan'],
            attention_mask=batch['attention_mask_plan']
        )['pooler_output']
        concat_pooler = torch.cat([assess_pooler_output, plan_pooler_output],
                                  dim=1)
        features = self.cls_dropout(concat_pooler)
        for l in self.classifier:
            features = l(features)
        return features


class BertSentenceSimilarity2(BertBase):
    """A sentence similarity classification model for 2022 n2c2 Track 3

    In this version, the assessment and plan are fed to BERT models that do not
    share weights."""
    def __init__(self, bert_name, num_cls_layers):
        super().__init__(bert_name)

        # Load the 2nd BERT module
        print(f'Load the 2nd BERT module {bert_name}... ', end='')
        self.bert2 = AutoModel.from_pretrained(bert_name)
        print('done')

        # Define the classification header
        self.cls_dropout = nn.Dropout(p=0.1)
        cls_layers = []
        for i in range(num_cls_layers):
            if i == 0:
                in_features = 2 * self.bert.config.hidden_size
                out_features = self.bert.config.hidden_size
            elif i != num_cls_layers - 1:
                in_features = out_features = self.bert.config.hidden_size
            else:
                in_features, out_features = self.bert.config.hidden_size, 4
            cls_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            if i != num_cls_layers - 1:
                cls_layers.append(nn.ReLU())
        self.classifier = nn.ModuleList(cls_layers)

    def get_param_group(self, train_bert, finetune_factor):
        if not train_bert:
            param_group = [{'params': list(self.parameters()), 'lr_factor': 1.0}]
        else:
            bert_params = list(self.bert.parameters()) + list(self.bert2.parameters())
            bert_params_set = set(bert_params)
            other_params = [p for p in list(self.parameters()) if p not in bert_params_set]
            param_group = [{'params': other_params, 'lr_factor': 1.0},
                           {'params': bert_params, 'lr_factor': finetune_factor}]
        return param_group

    def forward(self, batch):
        assess_pooler_output = self.bert(
            input_ids=batch['input_ids_assessment'],
            attention_mask=batch['attention_mask_assessment']
        )['pooler_output']
        plan_pooler_output = self.bert2(
            input_ids=batch['input_ids_plan'],
            attention_mask=batch['attention_mask_plan']
        )['pooler_output']
        concat_pooler = torch.cat([assess_pooler_output, plan_pooler_output],
                                  dim=1)
        features = self.cls_dropout(concat_pooler)
        for l in self.classifier:
            features = l(features)
        return features
