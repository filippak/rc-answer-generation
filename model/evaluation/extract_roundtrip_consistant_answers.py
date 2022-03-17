import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eval import get_token_segments

CRA_TOKENS =  ['[BGN]', '[END]']
# convert_tokens_to_ids -- to get id of tokens!! 
# use this to remove BGN och END from CRA encodings.

def get_CRA_map(CRA_data):
    context_text_map = {}
    for data in CRA_data:
        id = data['context_id']
        if id in context_text_map:
            context_text_map[id].append(data)
        else:
            context_text_map[id] = [data]
    return context_text_map

def get_tokens_for_segments(data, segments, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(data["input_ids"])
    all_l = []
    for segment in segments:
        # print('segment: ', segment)
        l = ''
        for i in range(segment[1]):
            token = tokens[segment[0]+i]
            # token can be NoneType?
            if token.startswith('##'):
                l += token[2:]
            elif len(l) > 0: # not the first word in answer phrase
                l += ' ' + token
            else:
                l += token
        # print(l)
        all_l.append(l)
    return all_l

def get_answers(data, tokenizer, answer_type):
    tokens = tokenizer.convert_ids_to_tokens(data["input_ids"])
    segments = get_token_segments(data[answer_type])
    all_l = []
    for segment in segments:
        # print('segment: ', segment)
        l = ''
        for i in range(segment[1]):
            token = tokens[segment[0]+i]
            # token can be NoneType?
            if token.startswith('##'):
                l += token[2:]
            elif len(l) > 0: # not the first word in answer phrase
                l += ' ' + token
            else:
                l += token
        # print(l)
        all_l.append(l)
    return all_l

def remove_BGN_END_tokens(CRA_data, tokenizer):
    ids_to_remove = tokenizer.convert_tokens_to_ids(CRA_TOKENS)
    mod_input_ids = []
    mod_labels = []
    mod_preds = []
    for idx, tok_id in enumerate(CRA_data['input_ids']):
        if not tok_id in ids_to_remove:
            mod_input_ids.append(tok_id)
            mod_labels.append(CRA_data['true_labels'][idx])
            mod_preds.append(CRA_data['predicted_labels'][idx])
    CRA_data['input_ids'] = mod_input_ids
    CRA_data['true_labels'] = mod_labels
    CRA_data['predicted_labels'] = mod_preds
    return CRA_data

def make_CRA_seg_str(segments):
    keys = []
    for s in segments:
        key = str(s[0]) + ' ' + str(s[1])
        if not key in keys:
            keys.append(key)
    return keys


def compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map):
    num_removed = 0
    total_num_predicted_answers = 0
    total_num_answers = 0
    for data in CA_data:
        segments = get_token_segments(data['predicted_labels'])
        context_id = data['context_id']
        CRA_data_context = context_text_map[context_id]
        CRA_segments = []
        # collect all CRA segments!
        for CRA_data in CRA_data_context:
            data_mod = remove_BGN_END_tokens(CRA_data, tokenizer)
            CRA_segments += get_token_segments(data_mod['predicted_labels'])
        
        label_answers = get_answers(data, tokenizer, 'true_labels')
        total_num_answers += len(label_answers)
        print('correct_answers: ', label_answers)
        CRA_keys = make_CRA_seg_str(CRA_segments)
        ok_answers = []
        for s in segments:
            key = str(s[0]) + ' ' + str(s[1])
            if key in CRA_keys:
                ok_answers.append(s)
                total_num_predicted_answers += 1
            else:
                num_removed += 1

        ans = get_tokens_for_segments(data, ok_answers, tokenizer)
        print(ans)

    print('number of predicted: ', total_num_predicted_answers)
    print('number of removed: ', num_removed)
    print('number of answers: ', total_num_answers)

def compare_text_segments(CA_data, CRA_data, tokenizer, context_text_map):
    num_removed = 0
    total_num_predicted_answers = 0
    total_num_answers = 0
    for data in CA_data:
        CA_answers = get_answers(data, tokenizer, 'predicted_labels')
        context_id = data['context_id']
        CRA_data_context = context_text_map[context_id]
        CRA_answers = []
        # collect all CRA answers!
        for CRA_data in CRA_data_context:
            CRA_answers += get_answers(CRA_data, tokenizer, 'predicted_labels')
        label_answers = get_answers(data, tokenizer, 'true_labels')
        total_num_answers += len(label_answers)
        print('correct_answers: ', label_answers)
        consistent_answers = []
        for ans in CA_answers:
            total_num_predicted_answers += 1
            if ans in CRA_answers:
                print('answer: ', ans)
                consistent_answers.append(ans)
            else:
                num_removed += 1
    print('number of predicted: ', total_num_predicted_answers)
    print('number of removed: ', num_removed)
    print('number of answers: ', total_num_answers)

def compare_CA_CRA_predictions(CA_data, CRA_data, tokenizer):
    # get the predicted labels for each context text
    context_text_map = get_CRA_map(CRA_data)
    print('len CRA map: ', len(context_text_map.keys()))
    compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map)
    # compare_text_segments(CA_data, CRA_data, tokenizer, context_text_map)
    



def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
    print('Added ', num_added_toks, 'tokens')

    with open(args.CA_data_path, "rb") as input_file:
        CA_data = pickle.load(input_file)

    with open(args.CRA_data_path, "rb") as input_file:
        CRA_data = pickle.load(input_file)

    compare_CA_CRA_predictions(CA_data, CRA_data, tokenizer)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('CA_data_path', type=str, 
        help='path CA output data', action='store')
    parser.add_argument('CRA_data_path', type=str, 
        help='path to CRA data file', action='store')

    args = parser.parse_args()
    main(args)