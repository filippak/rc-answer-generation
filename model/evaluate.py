# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import json
import torch

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

SPECIAL_TOKEN_LABEL = -100

labels = [
    "0",
    "B-answer",
    "I-answer",
    "B-I-answer",
]

def load_data():
    df = pd.read_pickle("../data-analysis/create_dataset/labeled_training_data.pkl")
    return df

def tokenize_data(tokenizer, data):
    contexts = data['tokens'].tolist()
    labels = data['labels'].tolist()
    delim = np.floor(len(contexts)*0.2)

    context_train = contexts[:int(len(contexts)-delim)]
    context_val = contexts[int(len(contexts)-delim):]
    labels_train = labels[:int(len(contexts)-delim)]
    labels_val = labels[int(len(contexts)-delim):]
    tokenized_inputs_train = align_labels(context_train, labels_train)
    tokenized_inputs_val = align_labels(context_val, labels_val)
    
    train_arr = []
    val_arr = []

    for idx, item in data.iterrows():
        context = item['tokens']
        labels = item['labels']
        # tokenized_input = tokenizer(context, truncation=True, max_length=512, is_split_into_words=True)
        
        tokenized_input, tokens_raw, labels_raw = align_labels_single_context(context, labels)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
 
        # TODO: use train/test splitter!
        if idx < delim:
            val_arr.append(tokenized_input)
        else:
            train_arr.append(tokenized_input)

    return tokenized_inputs_train, tokenized_inputs_val, train_arr, val_arr

def align_labels_single_context(context, labels_in):
    tokenized_inputs = tokenizer(context, truncation=True, max_length=512, is_split_into_words=True)


    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(SPECIAL_TOKEN_LABEL)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(labels_in[word_idx])
        else:
            label_ids.append(SPECIAL_TOKEN_LABEL)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    tokens = tokenized_inputs['input_ids']
    return tokenized_inputs, tokens, label_ids

def align_labels(context, labels_in):
    # function from https://huggingface.co/docs/transformers/custom_datasets
    # tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tokenized_inputs = tokenizer(context, truncation=True, max_length=512, is_split_into_words=True)

    labels = []
    for i, label in enumerate(labels_in):
        # print('label length PRE: ', len(label))
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(SPECIAL_TOKEN_LABEL)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(SPECIAL_TOKEN_LABEL)
            previous_word_idx = word_idx
        labels.append(label_ids)
        # print('label ids length: ', len(label_ids))
        # print('token length: ', len(word_ids))
    print('tokens: ', len(tokenized_inputs['input_ids']))
    tokenized_inputs["labels"] = labels
    print('labels: ', len(tokenized_inputs['labels']))
    return tokenized_inputs


if __name__ == '__main__':
    data = load_data()
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

    tokenized_inputs_train, tokenized_inputs_val, train_arr, val_arr = tokenize_data(tokenizer, data)
    # print(tokenized_inputs_train)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    model = torch.load('./results/model_file')
    # model = AutoModelForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=len(labels))

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_arr,
        eval_dataset=val_arr,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    # print('training model..')
    # trainer.train()
    print('Starting evaluation')
    trainer.evaluate()
    print('finished evaluation')

    # torch.save(model, './results/model_file')
    model.eval()



