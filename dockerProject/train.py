# file to load the data, tokenize and update labels accordingly
import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments
import torch
import pickle
from helper import ContextAnswerDataset, WeightedLossTrainerCA, WeightedLossTrainerCAR, WeightedLossTrainerCRA
import argparse
import random

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


def main(args):
    print("Is Cuda Available: ", torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("Num GPUs Avalailable: ", torch.cuda.device_count())
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data, val_data = load_data(args.data_path)
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    data_collator = DataCollatorForTokenClassification(tokenizer)
    num_labels = args.num_labels

     # Linear layer on top of the hidden states output
    # Implementation details in: 
    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py
    # row 1693 -> 
    model = AutoModelForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=num_labels).to(device)
    print('model device: ', model.device)

    if args.CRA:
        num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
        print('Added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))

    train_data, val_data = make_batches(train_data, val_data)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps", # can be epochs, then add logging_strategy="epoch",
        eval_steps=20,
        logging_steps=20,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True
    )
    print('Training args device: ', training_args.device)
    
    if args.CAR:
        trainer = WeightedLossTrainerCAR(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.CRA:
        trainer = WeightedLossTrainerCRA(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        trainer = WeightedLossTrainerCA(
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

    trainer.save_model(args.output_path)

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

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    main(args)



