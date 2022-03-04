# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import json
import torch
import pickle
from sklearn.model_selection import train_test_split
import wandb

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

# https://wandb.ai/filippak/answer-extraction

wandb.init(project="answer-extraction", entity="filippak")

SPECIAL_TOKEN_LABEL = -100

labels = [
    "0",
    "B-answer",
    "I-answer",
]

def load_data():
    with open(r'./data/tokenized_data_arr.pkl', "rb") as input_file:
        data = pickle.load(input_file)
    return data

def make_batches(data):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print('Length of training data', len(train_data))
    print('Length of validation data', len(val_data))
    return train_data, val_data


if __name__ == '__main__':
    data = load_data()
    # data is already tokenized with tokenizeer in the dataset.py script
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

    data_collator = DataCollatorForTokenClassification(tokenizer)
    model = AutoModelForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=len(labels))

    train_data, val_data = make_batches(data)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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

    torch.save(model, './results/model_file_3_labels')
    torch.save(model, './results/model_file_3_labels.pkl')
    # model.eval()



