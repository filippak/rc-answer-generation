# file to load the data, tokenize and update labels accordingly
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, ElectraTokenizer
import pickle
import argparse
import random
import copy

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# https://huggingface.co/docs/transformers/custom_datasets

# For adding context and answer to same tokenized object 
# https://huggingface.co/transformers/v3.2.0/glossary.html

SPECIAL_TOKEN_LABEL = -100
# max lengths creating C,A -> R data encoding
MAX_TOK_LEN = 448
MAX_ANS_LEN = 64
MAX_LEN = 512


CRA_TOKENS =  ['[BGN]', '[END]']

def load_data(path):
    df = pd.read_pickle(path)
    return df

def tokenize_data(tokenizer, data):
    tokenized_inputs_arr = []
    tokenized_inputs_arr_with_id = []
    special_token_ids = tokenizer.convert_tokens_to_ids(CRA_TOKENS)
    num_removed = 0
    print(special_token_ids)
    num_neg_for_text = 0
    for idx, item in data.iterrows():
        context = item['tokens']
        ans = item['answer']
        sent = item['sentence']
        label = item['label'] # 0 / 1
        # calculating the max length..
        tokenized_sent_ans = tokenizer(sent,ans, truncation=True, max_length=MAX_TOK_LEN, is_split_into_words=True)
        # TODO: fix this so that this can be run with both (CSA) and (CA) modes
        c_len = MAX_LEN - len(tokenized_sent_ans['input_ids'])
        tokenized_input = tokenizer([context, sent, ans], truncation=True, max_length=c_len, is_split_into_words=True)
        # remove CLS token
        tokenized_input["input_ids"][1] = tokenized_input["input_ids"][1][1:] # remove first CLS
        tokenized_input["input_ids"][2] = tokenized_input["input_ids"][2][1:] # remove first CLS
        tokenized_input["input_ids"] = [item for sublist in tokenized_input["input_ids"] for item in sublist]
        tokenized_input["attention_mask"][1] = tokenized_input["attention_mask"][1][1:] # remove first CLS
        tokenized_input["attention_mask"][2] = tokenized_input["attention_mask"][2][1:] # remove first CLS
        tokenized_input["attention_mask"] = [item for sublist in tokenized_input["attention_mask"] for item in sublist]
        tokenized_input["token_type_ids"][1] = tokenized_input["token_type_ids"][1][1:] # remove first CLS
        tokenized_input["token_type_ids"][2] = tokenized_input["token_type_ids"][2][1:] # remove first CLS
        token_type_ids_sent = [x+1 for x in tokenized_input["token_type_ids"][1]]
        token_type_ids_ans = [x+1 for x in tokenized_input["token_type_ids"][2]]
        tokenized_input["token_type_ids"][1] = token_type_ids_sent
        tokenized_input["token_type_ids"][2] = token_type_ids_ans
        tokenized_input["token_type_ids"] = [item for sublist in tokenized_input["token_type_ids"] for item in sublist]
        if len(tokenized_input["input_ids"]) > 512:
            print('TOO long!!!')
        # check  that the tokenized input includes [BGN] and [END] tokens, only add if it does
        if special_token_ids[0] in tokenized_input["input_ids"] and special_token_ids[1] in tokenized_input["input_ids"]:
            tokenized_input['label'] = label
            tokenized_inputs_arr.append(tokenized_input)
            # create copy with context id (does not work to train on)
            tokenized_input_with_id = copy.deepcopy(tokenized_input)
            tokenized_input_with_id['context_id'] = item['context_id'] # set the context id of tokenized data
            tokenized_inputs_arr_with_id.append(tokenized_input_with_id)
        else:
            # print('DOES NOT INCLUDE IDS!')
            # print('input: ', tokenized_input)
            # decoded = tokenizer.decode(tokenized_input["input_ids"])
            # print('decoded: ', decoded)
            num_removed += 1
    print('num removed: ', num_removed)
    print('Example: ', tokenized_inputs_arr[20])
    print('Example: ', tokenized_inputs_arr[21])
    return tokenized_inputs_arr, tokenized_inputs_arr_with_id



def main(args):
    data = load_data(args.data_path)
    if args.electra:
        print('Using electra')
        tokenizer = ElectraTokenizer.from_pretrained('KB/electra-base-swedish-cased-discriminator')
    else:
        tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    # Add BGN and END tokens
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
    print('Added ', num_added_toks, 'tokens')
    print('Tokenizing data..')
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
    parser = argparse.ArgumentParser(description='Prepare dataset for sentence classification with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--electra', dest='electra', action='store_true')

    args = parser.parse_args()
    main(args)
