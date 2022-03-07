# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import pickle
from sklearn.model_selection import train_test_split
import wandb
from torch.utils.data import DataLoader
from helper import ContextAnswerDataset
import argparse
import random

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

# https://wandb.ai/filippak/answer-extraction

SPECIAL_TOKEN_LABEL = -100
BATCH_SIZE = 16


def load_data(path):
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
    return data

def make_batches(data, seed):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=seed)
    # put data in custom dataset class
    train_dataset = ContextAnswerDataset(train_data)
    val_dataset = ContextAnswerDataset(val_data)

    print('Length of training data', len(train_data))
    print('Length of validation data', len(val_data))
    return train_dataset, val_dataset

def main(args):
    wandb.init(project=args.wandb_project, entity="filippak")
    data = load_data(args.data_path)
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

    data_collator = DataCollatorForTokenClassification(tokenizer)
    num_labels = args.num_labels
    model = AutoModelForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=num_labels)

    train_data, val_data = make_batches(data, args.seed)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print('training model..')
    trainer.train()
    print('finished training model')
    trainer.evaluate()
    print('finished evaluation')

    torch.save(model, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune bert model for token classification')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('wandb_project', type=str,
        help='wandb project name (can be answer-extraction or sentence-extraction)', action='store')
    parser.add_argument('num_labels', type=int, 
        help='number of labels', action='store', default=3)
    parser.add_argument('epochs', type=int, 
        help='number of training epochs', action='store', default=3)
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)



