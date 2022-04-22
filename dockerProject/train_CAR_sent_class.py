# file to load the data, tokenize and update labels accordingly
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
import pickle
from helper import ContextAnswerDataset, WeightedLossTrainerCARSentClass
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
    print("Num GPUs Avalailable: ", torch.cuda.device_count())

    train_data, val_data = load_data(args.data_path)
    
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    num_labels = args.num_labels

    # Linear layer on top of pooled output (= the output for the [CLS] token?)
    # Implementation details in: 
    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py
    # row 1501 -> 
    model = AutoModelForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=num_labels).to("cuda")
    print('model device: ', model.device)
    
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
    print('Added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))

    train_data, val_data = make_batches(train_data, val_data)

    training_args = TrainingArguments(
        output_dir="./results",
        # evaluation_strategy="epoch", # can be epochs, then add logging_strategy="epoch",
        # logging_strategy="epoch",
        save_steps=2000,
        evaluation_strategy="steps", # can be epochs, then add logging_strategy="epoch",
        eval_steps=2000,
        logging_steps=2000,
        save_total_limit=5,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True
    )
    print('Training args device: ', training_args.device)

    trainer = WeightedLossTrainerCARSentClass(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )
    print('training model..')
    trainer.train()
    print('finished training model')
    trainer.evaluate()
    print('finished evaluation')

    trainer.save_model(args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune bert model for sequence classification')

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
    parser.add_argument('--save', dest='save_data', action='store_true')
    parser.add_argument('--num_train', type=int, dest='num_train', action='store', default=2000)
    parser.add_argument('--num_val', type=int, dest='num_val', action='store', default=400)
    parser.add_argument('--save_path', dest='save_path', action='store')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    main(args)



