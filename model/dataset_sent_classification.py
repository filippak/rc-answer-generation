# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import pickle
import argparse
import random
import copy

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

SPECIAL_TOKEN_LABEL = -100
# max lengths creating C,A -> R data encoding
MAX_TOK_LEN = 448
MAX_ANS_LEN = 64

labels = [
    "0",
    "1",
    "I-answer",
]
CRA_TOKENS =  ['[BGN]', '[END]']

def load_data(path):
    df = pd.read_pickle(path)
    return df

def tokenize_data(tokenizer, data):
    tokenized_inputs_arr = []
    tokenized_inputs_arr_with_id = []
    for idx, item in data.iterrows():
        context = item['tokens']
        label = item['label'] # 0 / 1
        tokenized_input = tokenizer(context, truncation=True, max_length=MAX_TOK_LEN, is_split_into_words=True, add_special_tokens=True)
        tokenized_input['label'] = label
        tokenized_inputs_arr.append(tokenized_input)
        # create copy with context id (does not work to train on)
        tokenized_input_with_id = copy.deepcopy(tokenized_input)
        tokenized_input_with_id['context_id'] = item['context_id'] # set the context id of tokenized data
        tokenized_inputs_arr_with_id.append(tokenized_input_with_id)

    return tokenized_inputs_arr, tokenized_inputs_arr_with_id

def parse_answers(data, tokenizer):
    ans_arr = []

    for idx, item in data.iterrows():
        ans = item['answer'] # array of tokens

        # Don't add special tokens to the tokenized answer! (removes [CLS] in the beginning and [SEP] in the end)
        tokenized_ans = tokenizer(ans, truncation=True, max_length=MAX_ANS_LEN, is_split_into_words=True, add_special_tokens=False)
        ans_arr.append(tokenized_ans)
    
    return ans_arr

def add_answers_to_tokenized_data(tokenized_context, ans_data, tokenizer):
    for idx, context in enumerate(tokenized_context):
        # context is an object with keys: input_ids, token_type_ids, attention_mask, label
        ans = ans_data[idx] # array of each of the answers in the dataset 
        context['input_ids'] += ans['input_ids']
        context['token_type_ids'] += ans['token_type_ids']
        context['attention_mask'] += ans['attention_mask']

    return tokenized_context



def main(args):
    data = load_data(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    # Add BGN and END tokens
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
    print('Added ', num_added_toks, 'tokens')

    tokenized_inputs_arr, tokenized_inputs_arr_with_id = tokenize_data(tokenizer, data)
    ans_tokenized_inputs_arr = parse_answers(data, tokenizer) # array of the tokenized answer data
    tokenized_inputs_arr = add_answers_to_tokenized_data(tokenized_inputs_arr, ans_tokenized_inputs_arr, tokenizer)
    tokenized_inputs_arr_with_id = add_answers_to_tokenized_data(tokenized_inputs_arr_with_id, ans_tokenized_inputs_arr, tokenizer)

    # save the train and eval data
    print('Data size: ', len(tokenized_inputs_arr))
    model_data_path = args.output_path + '.pkl'
    eval_data_path = args.output_path + '_with_id.pkl'
    with open(model_data_path, "wb") as output_file:
        pickle.dump(tokenized_inputs_arr, output_file)

    with open(eval_data_path, "wb") as output_file:
        pickle.dump(tokenized_inputs_arr_with_id, output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for sentence classification with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')

    args = parser.parse_args()
    main(args)
