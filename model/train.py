# file to load the data, tokenize and update labels accordingly
import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments
import torch
import pickle
import wandb
from helper import ContextAnswerDataset, WeightedLossTrainerCA, WeightedLossTrainerCAR, WeightedLossTrainerCRA
import argparse
import random
from datasets import load_metric

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=FBiW8UpKIrJW

# https://wandb.ai/filippak/answer-extraction

BATCH_SIZE = 8
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

def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main(args):
    wandb.init(project=args.wandb_project, entity="filippak")
    train_data, val_data = load_data(args.data_path)
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    data_collator = DataCollatorForTokenClassification(tokenizer)
    num_labels = args.num_labels

     # Linear layer on top of the hidden states output
    # Implementation details in: 
    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py
    # row 1693 -> 
    model = AutoModelForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=num_labels)
    if args.CRA:
        num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
        print('Added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))

    train_data, val_data = make_batches(train_data, val_data)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps", # can be epochs, then add logging_strategy="epoch",
        eval_steps=10,
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="wandb",
        load_best_model_at_end=True
    )
    if args.CAR:
        trainer = WeightedLossTrainerCAR(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    elif args.CRA:
        trainer = WeightedLossTrainerCRA(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = WeightedLossTrainerCA(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
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
    parser.add_argument('--CAR', dest='CAR', action='store_true')
    parser.add_argument('--CRA', dest='CRA', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    # torch manual seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    main(args)



