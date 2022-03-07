# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

import pickle
import argparse
import random

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

SPECIAL_TOKEN_LABEL = -100
# max lengths creating C,A -> R data encoding
MAX_TOK_LEN = 448
MAX_ANS_LEN = 64

labels = [
    "0",
    "B-answer",
    "I-answer",
]

labels_s = [
    "0",
    "B-sentence",
    "I-sentence",
]

def load_data(path):
    df = pd.read_pickle(path)
    return df

def tokenize_data(tokenizer, data, max_tok_len=512):
    contexts = data['tokens'].tolist()
    labels = data['labels'].tolist()

    tokenized_inputs = align_labels(contexts, labels, tokenizer, max_tok_len)
    
    tokenized_inputs_arr = []

    for idx, item in data.iterrows():
        context = item['tokens']
        labels = item['labels']
        
        tokenized_input, tokens_raw, labels_raw = align_labels_single_context(context, labels, tokenizer, max_tok_len)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
 
        tokenized_inputs_arr.append(tokenized_input)

    return tokenized_inputs, tokenized_inputs_arr

def align_labels_single_context(context, labels_in, tokenizer, max_tok_len=512):
    tokenized_inputs = tokenizer(context, truncation=True, max_length=max_tok_len, is_split_into_words=True)


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

def align_labels(context, labels_in, tokenizer, max_tok_len):
    # function from https://huggingface.co/docs/transformers/custom_datasets
    # tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tokenized_inputs = tokenizer(context, truncation=True, max_length=max_tok_len, is_split_into_words=True)

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
    print('tokens: ', len(tokenized_inputs['input_ids']))
    tokenized_inputs["labels"] = labels
    print('labels: ', len(tokenized_inputs['labels']))
    return tokenized_inputs

def parse_answers(data, tokenizer, max_ans_len):
    ans_tokenized_inputs_arr = []

    for idx, item in data.iterrows():
        ans_arr = item['answers']
        ans_labels_arr = item['answer_labels']
        ans_len = 0
        ans_for_single_context = []
        for a_idx, ans in enumerate(ans_arr):
            ans_labels = ans_labels_arr[a_idx]
            ans_len += len(ans_labels)
            # only add the answer if there is space in the embedding!! - is this an ok way??
            if ans_len <= max_ans_len:
                ans_tokenized_input, tokens_raw, labels_raw = align_labels_single_context(ans, ans_labels, tokenizer)
                tokens = tokenizer.convert_ids_to_tokens(ans_tokenized_input["input_ids"])
                # print('tokens: ', tokens)
                ans_for_single_context.append(ans_tokenized_input)
                # print('ans context: ', ans_for_single_context)

        ans_tokenized_inputs_arr.append(ans_for_single_context)
    
    return ans_tokenized_inputs_arr

def add_answers_to_tokenized_data(tokenized_context, ans_data, tokenizer):
    for idx, context in enumerate(tokenized_context):
        # context is an object with keys: input_ids, token_type_ids, attention_mask, labels
        ans_arr = ans_data[idx]
        for ans in ans_arr:
            context['input_ids'] += ans['input_ids']
            context['token_type_ids'] += ans['token_type_ids']
            context['attention_mask'] += ans['attention_mask']
            context['labels'] += ans['labels']
        # tokens = tokenizer.convert_ids_to_tokens(context["input_ids"])
        # print(tokens)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_context[0]["input_ids"])
    print(tokens)
    return tokenized_context



def main(args):
    data = load_data(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    max_tok_len = 512
    if args.parse_answers:
        max_tok_len = MAX_TOK_LEN
    # parse the training data
    tokenized_inputs, tokenized_inputs_arr = tokenize_data(tokenizer, data, max_tok_len)

    if args.parse_answers:
        ans_tokenized_inputs_arr = parse_answers(data, tokenizer, MAX_ANS_LEN)
        tokenized_inputs_arr = add_answers_to_tokenized_data(tokenized_inputs_arr, ans_tokenized_inputs_arr, tokenizer)
    
    # save the train and eval data
    arr_file = args.output_path + "_arr.pkl"
    obj_file = args.output_path + "_obj.pkl"

    with open(obj_file, "wb") as output_file:
        pickle.dump(tokenized_inputs, output_file)

    with open(arr_file, "wb") as output_file:
        pickle.dump(tokenized_inputs_arr, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--answers', dest='parse_answers', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=1)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
