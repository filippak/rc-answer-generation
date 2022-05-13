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
    "B-answer",
    "I-answer",
]

labels_s = [
    "0",
    "B-sentence",
    "I-sentence",
]

CRA_TOKENS =  ['[BGN]', '[END]']

def load_data(path):
    df = pd.read_pickle(path)
    return df

def tokenize_data(tokenizer, data, max_tok_len=512, CRA_tok_ids=None):
    num_max_len = 0
    tokenized_inputs_arr = []
    tokenized_inputs_arr_with_id = []
    for idx, item in data.iterrows():
        context = item['tokens']
        labels = item['labels']
        tokenized_input = align_labels_single_context(context, labels, tokenizer, max_tok_len, True, CRA_tok_ids)
        if len(tokenized_input['input_ids']) == 512:
            num_max_len += 1
        tokenized_inputs_arr.append(tokenized_input)
        # create copy with context id (does not work to train on)
        tokenized_input_with_id = copy.deepcopy(tokenized_input)
        tokenized_input_with_id['context_id'] = item['context_id'] # set the context id of tokenized data
        tokenized_inputs_arr_with_id.append(tokenized_input_with_id)
    print('Input of max len: ', num_max_len)
    return tokenized_inputs_arr, tokenized_inputs_arr_with_id

def align_labels_single_context(context, labels_in, tokenizer, max_tok_len=512, special_tokens=True, CRA_tok_ids=None):
    tokenized_inputs = tokenizer(context, truncation=True, max_length=max_tok_len, is_split_into_words=True, add_special_tokens=special_tokens)

    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for idx, word_idx in enumerate(word_ids):  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(SPECIAL_TOKEN_LABEL)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            word_id = tokenized_inputs['input_ids'][idx] # get the word id of the current word
            if CRA_tok_ids and word_id in CRA_tok_ids: # if want to set BGN and END to -100 (otherwise 0)
                label_ids.append(SPECIAL_TOKEN_LABEL)
            else:
                label_ids.append(labels_in[word_idx])
        else:
            label_ids.append(SPECIAL_TOKEN_LABEL)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

def parse_answers(data, tokenizer):
    ans_arr = []

    for idx, item in data.iterrows():
        ans = item['answer'] # array of tokens
        ans_labels = item['answer_labels'] # array of labels

        # Don't add special tokens to the tokenized answer! (removes [CLS] in the beginning and [SEP] in the end)
        ans_tokenized = align_labels_single_context(ans, ans_labels, tokenizer, MAX_ANS_LEN, False)
        ans_arr.append(ans_tokenized)
    
    return ans_arr

def add_answers_to_tokenized_data(tokenized_context, ans_data, tokenizer):
    for idx, context in enumerate(tokenized_context):
        # context is an object with keys: input_ids, token_type_ids, attention_mask, labels
        ans = ans_data[idx] # array of each of the answers in the dataset 
        context['input_ids'] += ans['input_ids']
        context['token_type_ids'] += ans['token_type_ids']
        context['attention_mask'] += ans['attention_mask']
        context['labels'] += ans['labels']

    return tokenized_context



def main(args):
    data = load_data(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    if args.parse_answers:
        tokenized_inputs_arr, tokenized_inputs_arr_with_id = tokenize_data(tokenizer, data, MAX_TOK_LEN)
        ans_tokenized_inputs_arr = parse_answers(data, tokenizer) # array of the tokenized answer data
        tokenized_inputs_arr = add_answers_to_tokenized_data(tokenized_inputs_arr, ans_tokenized_inputs_arr, tokenizer)
        tokenized_inputs_arr_with_id = add_answers_to_tokenized_data(tokenized_inputs_arr_with_id, ans_tokenized_inputs_arr, tokenizer)
    else:
        if args.CRA:
            num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
            print('Added ', num_added_toks, 'tokens')
            if args.CRA_tok_ignore:
                special_token_ids = tokenizer.convert_tokens_to_ids(CRA_TOKENS)
                tokenized_inputs_arr, tokenized_inputs_arr_with_id = tokenize_data(tokenizer, data, 512, special_token_ids)
            else:
                tokenized_inputs_arr, tokenized_inputs_arr_with_id = tokenize_data(tokenizer, data)
        else:
            tokenized_inputs_arr, tokenized_inputs_arr_with_id = tokenize_data(tokenizer, data)
    
    # save the train and eval data
    print('Data size: ', len(tokenized_inputs_arr))
    model_data_path = args.output_path + '.pkl'
    eval_data_path = args.output_path + '_with_id.pkl'
    with open(model_data_path, "wb") as output_file:
        pickle.dump(tokenized_inputs_arr, output_file)

    with open(eval_data_path, "wb") as output_file:
        pickle.dump(tokenized_inputs_arr_with_id, output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--answers', dest='parse_answers', action='store_true')
    parser.add_argument('--CRA', dest='CRA', action='store_true')
    parser.add_argument('--CRA_tok_ignore', dest='CRA_tok_ignore', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
