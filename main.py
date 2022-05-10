#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os
import pickle
import sys
import time
import types

import fairseq
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import get_dataset
from src.models import get_model
from src.utils.checkpoint_manager import CheckPointManager


parser = argparse.ArgumentParser(description="Train and test the model")
# Dataset
parser.add_argument("--dataset", type=str, default="relation_dataset")
parser.add_argument("--data_dir", type=str, default="data/N2C2-Track3-May3")
parser.add_argument("--train_file", type=str, default="train.csv")
parser.add_argument("--dev_file", type=str, default="dev.csv")
parser.add_argument("--tokenizer_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
parser.add_argument("--max_len", type=int, default=512)
# Model
parser.add_argument("--model", type=str, default="bert_sent_rel")
parser.add_argument("--bert_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
parser.add_argument("--num_cls_layers", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="results/bert_sim")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument('--gpu', default=False, action='store_true')
# Optimization
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument('--train_bert', default=False, action='store_true')
parser.add_argument("--bert_finetune_factor", type=float, default=0.1)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--adam_betas", type=str, default='(0.9, 0.999)')
parser.add_argument("--adam_eps", type=float, default=1e-08)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--training_step", type=int, default=10000)
parser.add_argument("--display_iter", type=int, default=25)
parser.add_argument("--eval_iter", type=int, default=1000)
parser.add_argument("--lr_scheduler", type=str, default='inverse_sqrt')
parser.add_argument("--lr", type=float, nargs='+', default=[0.0001])
parser.add_argument("--warmup_updates", type=int, default=10000)
parser.add_argument("--warmup_init_lr", type=float, default=1e-6)
parser.add_argument("--init_ckpt", type=str, default=None)
parser.add_argument("--init_step", type=int, default=0)
# Evaluation
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument("--test_file", type=str, default="test.csv")
args = parser.parse_args()

# Print parameters
print('\n'.join([f'\t{k}: {v}' for k, v in vars(args).items()]))


def set_lr(self, lr):
    """Setting lr on each param group (override on a fairseq optimizer) """
    for param_group in self.optimizer.param_groups:
        if 'lr_factor' in param_group:
            param_group['lr'] = lr * param_group['lr_factor']
        else:
            param_group['lr'] = lr


def get_epoch_results(model, test_dataloader):
    """Perform the model forward on the whole epoch."""
    results = []
    for batch_test in tqdm(test_dataloader):
        if args.gpu:
            for k in batch_test:
                batch_test[k] = batch_test[k].cuda()

        batch_logits = model(batch_test).cpu().numpy()
        for i in range(len(batch_logits)):
            results.append({
                'row_id': int(batch_test['row_id'][i]),
                'hadm_id': int(batch_test['hadm_id'][i]),
                'logits': batch_logits[i],
                'label': int(batch_test['label'][i])
            })
    return results


def main():
    # Set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset
    if not args.test:
        train_dataset = get_dataset(dataset=args.dataset,
                                    data_file=os.path.join(args.data_dir, args.train_file),
                                    tokenizer_name=args.tokenizer_name,
                                    max_len=args.max_len)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       num_workers=4,  # 4 * args.num_gpus
                                                       collate_fn=train_dataset.collate_fn,
                                                       persistent_workers=True)
        dev_dataset = get_dataset(dataset=args.dataset,
                                  data_file=os.path.join(args.data_dir, args.dev_file),
                                  tokenizer_name=args.tokenizer_name,
                                  max_len=args.max_len)
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     batch_size=args.batch_size,
                                                     num_workers=1,
                                                     collate_fn=dev_dataset.collate_fn,
                                                     persistent_workers=True)
    else:
        test_dataset = get_dataset(dataset=args.dataset,
                                   data_file=os.path.join(args.data_dir, args.test_file),
                                   tokenizer_name=args.tokenizer_name,
                                   max_len=args.max_len)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      batch_size=args.batch_size,
                                                      num_workers=4,
                                                      collate_fn=test_dataset.collate_fn)

    # Model
    model = get_model(model=args.model, bert_name=args.bert_name, num_cls_layers=args.num_cls_layers)

    # Checkpoint
    init_step = 0
    ckpt_manager = CheckPointManager(args.output_dir)
    if args.init_ckpt is not None:
        print(f'Load model ckpt from {args.init_ckpt} (step {args.init_step})')
        model.load_state_dict(torch.load(args.init_ckpt))
        init_step = args.init_step
    else:
        ckpt_status = ckpt_manager.get_latest_checkpoint()
        if ckpt_status:
            init_step, ckpt_fname = ckpt_status
            ckpt_fpath = os.path.join(args.output_dir, ckpt_fname)
            print(f'Load model ckpt from {ckpt_fpath} (step {init_step})')
            ckpt_manager.load_ckpt(model, ckpt_fname)
    global_step = init_step

    # Save arguments at first
    if not args.test and global_step == 0:
        ckpt_manager.save_args(args)

    # Set cuda
    if args.gpu:
        model.cuda()

    # Train / test
    if not args.test:
        # Objective function
        criteria = torch.nn.CrossEntropyLoss()

        # Optimizer and LR scheduler
        params = model.get_param_group(args.train_bert, args.bert_finetune_factor)
        optimizer = fairseq.optim.build_optimizer(args, params)
        optimizer.set_lr = types.MethodType(set_lr, optimizer) # Override the set_lr() of the optimizer
        lr_scheduler = fairseq.optim.lr_scheduler.build_lr_scheduler(args, optimizer)
        lr_scheduler.step_update(global_step)

        # Tensorboard writer
        writer = SummaryWriter(args.output_dir)

        # Training!
        model.train()
        train_iter = iter(train_dataloader)
        dev_iter = iter(dev_dataloader)
        while global_step < args.training_step or global_step == init_step:
            start_time = time.time()
            optimizer.zero_grad()

            # Data preparing
            try:
                batch_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch_train = next(train_iter)

            if args.gpu:
                for k in batch_train:
                    batch_train[k] = batch_train[k].cuda()

            # Model forward
            output_logits = model.forward(batch_train)
            loss = criteria(output_logits, target=batch_train['label'])

            # Model backward
            loss.backward()
            optimizer.step()

            duration = time.time() - start_time

            # Print the training progress per display_iter steps
            if global_step % args.display_iter == 0:
                examples_per_sec = args.batch_size / duration
                train_acc = np.mean(np.argmax(output_logits.detach().cpu().numpy(), axis=1) == batch_train['label'].cpu().numpy())
                print_str = f'{datetime.now()}: (train) step {global_step:7}, loss={float(loss.cpu()):.6f}, acc={train_acc:.4f}'
                writer.add_scalar('train/loss', float(loss.cpu()), global_step)
                writer.add_scalar('train/accuracy', train_acc, global_step)
                writer.add_scalar('train/learning_rate', optimizer.get_lr(), global_step)
                writer.flush()
                print_str += f', lr={optimizer.get_lr():g} ({examples_per_sec:.1f} it/s, {duration:.3f} s/batch)'
                print(print_str)

            # Evaluate on dev dataset + save model ckpt per eval_iter steps
            if (global_step % args.eval_iter == 0 and global_step > init_step) or global_step == args.training_step - 1:
                model.eval()

                with torch.no_grad():
                    dev_results = get_epoch_results(model, dev_dataloader)
                    dev_logits = np.array([r['logits'] for r in dev_results])
                    dev_labels = np.array([r['label'] for r in dev_results])
                    dev_loss = criteria(torch.tensor(dev_logits), torch.tensor(dev_labels))
                    dev_acc = np.mean(np.argmax(dev_logits, axis=1) == dev_labels)
                    print(f'{datetime.now()}: (dev)   step {global_step:7}, loss={dev_loss.cpu():.6f}, acc={dev_acc:.4f}')
                    writer.add_scalar('dev/loss', float(dev_loss.cpu()), global_step)
                    writer.add_scalar('dev/accuracy', dev_acc, global_step)
                    writer.flush()

                save_ckpt_fname = ckpt_manager.save_ckpt(model, global_step)
                save_ckpt_fpath = os.path.join(args.output_dir, save_ckpt_fname)
                print(f'Save checkpoint to {save_ckpt_fpath}')

                torch.cuda.empty_cache()

                model.train()

            global_step += 1
            lr_scheduler.step_update(global_step)
    else:
        # Objective function
        criteria = torch.nn.CrossEntropyLoss()

        # Evaluate on test dataset
        model.eval()
        with torch.no_grad():
            print('Evaluating...')
            test_results = get_epoch_results(model, test_dataloader)
            test_logits = np.array([r['logits'] for r in test_results])
            test_labels = np.array([r['label'] for r in test_results])
            test_loss = criteria(torch.tensor(test_logits), torch.tensor(test_labels))
            test_acc = np.mean(np.argmax(test_logits, axis=1) == test_labels)
            print(f'{datetime.now()}: (test)  step {global_step:7}, loss={test_loss.cpu():.6f}, acc={test_acc:.4f}')

        # Save the eval results
        test_name = os.path.splitext(args.test_file)[0]
        test_result_fname = f'results_{test_name}_{global_step}.pkl'
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        test_result_path = os.path.join(args.output_dir, test_result_fname)
        print(f'Save val results to {test_result_path}')
        with open(test_result_path, 'wb') as fd:
            pickle.dump(test_results, fd)


if __name__ == "__main__":
    main()
