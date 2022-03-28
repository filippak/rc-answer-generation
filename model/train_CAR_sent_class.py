# file to load the data, tokenize and update labels accordingly
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
import pickle
import wandb
from helper import ContextAnswerDataset, WeightedLossTrainerCARSentClass
import argparse
import random
from datasets import load_metric

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=FBiW8UpKIrJW

# https://wandb.ai/filippak/answer-extraction

BATCH_SIZE = 16
CRA_TOKENS =  ['[BGN]', '[END]']


def load_data(path):
    train_path = path + '_train.pkl'
    val_path = path + '_eval.pkl'
    with open(train_path, "rb") as input_file:
        train_data = pickle.load(input_file)
    with open(val_path, "rb") as input_file:
        val_data = pickle.load(input_file)
    return train_data, val_data

def make_batches(train_data, val_data):
    # put data in custom dataset class
    train_dataset = ContextAnswerDataset(train_data)
    val_dataset = ContextAnswerDataset(val_data)

    print('Length of training data', len(train_data))
    print('Length of validation data', len(val_data))
    return train_dataset, val_dataset

# TODO: fix this! not working currently..
def compute_metrics(eval_pred):
    metric = load_metric("seqeval")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main(args):
    wandb.init(project=args.wandb_project, entity="filippak")
    train_data, val_data = load_data(args.data_path)
    
    # TODO: train with all data..
    random.shuffle(train_data)
    random.shuffle(val_data)
    train_data = train_data[:5000]
    val_data = val_data[:100]
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    num_labels = args.num_labels

    # Linear layer on top of pooled output (= the output for the [CLS] token?)
    # Implementation details in: 
    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py
    # row 1501 -> 
    model = AutoModelForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=num_labels)
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
    print('Added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))

    train_data, val_data = make_batches(train_data, val_data)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps", # can be epochs, then add logging_strategy="epoch",
        eval_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="wandb"
    )
    trainer = WeightedLossTrainerCARSentClass(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
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
        help='number of labels', action='store', default=2)
    parser.add_argument('epochs', type=int, 
        help='number of training epochs', action='store', default=3)

    args = parser.parse_args()
    main(args)



