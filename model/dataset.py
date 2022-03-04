# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import json
import pickle

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

SPECIAL_TOKEN_LABEL = -100

labels = [
    "0",
    "B-answer",
    "I-answer",
]

def load_data():
    df = pd.read_pickle("../data-analysis/create_dataset/labeled_training_data.pkl")
    df_test = pd.read_pickle("../data-analysis/create_dataset/labeled_test_data.pkl")
    return df, df_test

def tokenize_data(tokenizer, data):
    contexts = data['tokens'].tolist()
    labels = data['labels'].tolist()

    tokenized_inputs = align_labels(contexts, labels)
    
    tokenized_inputs_arr = []

    for idx, item in data.iterrows():
        context = item['tokens']
        labels = item['labels']
        
        tokenized_input, tokens_raw, labels_raw = align_labels_single_context(context, labels)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
 
        tokenized_inputs_arr.append(tokenized_input)

    return tokenized_inputs, tokenized_inputs_arr

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
    train_data, test_data = load_data()
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    # parse the training data
    tokenized_inputs, tokenized_inputs_arr = tokenize_data(tokenizer, train_data)
    
    # save the train and eval data
    with open(r'./data/tokenized_train_data_obj.pkl', "wb") as output_file:
        pickle.dump(tokenized_inputs, output_file)

    with open(r'./data/tokenized_train_data_arr.pkl', "wb") as output_file:
        pickle.dump(tokenized_inputs_arr, output_file)
    
    # parse the test data
    tokenized_test_inputs, tokenized_test_inputs_arr = tokenize_data(tokenizer, test_data)
    
    with open(r'./data/tokenized_test_data_obj.pkl', "wb") as output_file:
        pickle.dump(tokenized_test_inputs, output_file)

    with open(r'./data/tokenized_test_data_arr.pkl', "wb") as output_file:
        pickle.dump(tokenized_test_inputs_arr, output_file)