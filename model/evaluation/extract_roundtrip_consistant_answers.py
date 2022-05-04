import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
import copy
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eval import get_token_segments, confusion_matrix_tokens, evaluate_model_answer_spans, get_jaccard_score

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
    word_ids = data.word_ids()
    segments = get_token_segments(data[answer_type], word_ids)
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

def label_CA_data_mod(data, ok_answers):
    """
    Function to label the CA data with only the rountrip consistent answers
    """
    new_data = copy.deepcopy(data)
    new_predictions = [0 for _ in range(len(data['predicted_labels']))]
    for ans in ok_answers:
        for i in range(ans[1]):
            if i == 0:
                new_predictions[ans[0]+i] = 1
            else:
                new_predictions[ans[0]+i] = 2
    new_data['predicted_labels'] = new_predictions
    return new_data

def compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map):
    num_removed_exact = 0
    num_predicted_answers_exact = 0
    num_removed = 0
    num_predicted_answers = 0
    total_num_answers = 0
    CA_data_mod = []
    y_labels = []
    y_preds  = []
    answer_stats = {'FP': 0, 'TP': 0, 'FN': 0, 'jaccard': [], 'overlap': []}
    for data in CA_data:
        word_ids = data.word_ids()
        segments = get_token_segments(data['predicted_labels'], word_ids)
        context_id = data['context_id']
        print('context text id: ', context_id)
        CRA_data_context = context_text_map[context_id]
        CRA_segments = []
        # collect all CRA segments!
        for CRA_data in CRA_data_context:
            data_mod = remove_BGN_END_tokens(CRA_data, tokenizer)
            CRA_word_ids = data_mod.word_ids()
            CRA_segments += get_token_segments(data_mod['predicted_labels'], CRA_word_ids)
        
        # label_answers = get_answers(data, tokenizer, 'true_labels')
        # total_num_answers += len(label_answers)
        # print('correct_answers: ', label_answers)
        CRA_keys = make_CRA_seg_str(CRA_segments)
        ok_answers = []
        ok_answers_exact = []
        for s in segments:
            start = s[0]
            end = s[0]+s[1]-1
            key = str(s[0]) + ' ' + str(s[1])
            added = False
            for s_cra in CRA_segments:
                start_cra = s_cra[0]
                end_cra = s_cra[0]+s_cra[1]-1
                if not added and start <= end_cra and end >= start_cra:
                    #there is overlap!
                    # jacc = get_jaccard_score(s, s_cra)
                    # if jacc > 0.5:
                    ok_answers.append(s)
                    added = True
                    num_predicted_answers += 1
            if not added:
                num_removed += 1

            if key in CRA_keys:
                ok_answers_exact.append(s)
                num_predicted_answers_exact += 1
            else:
                num_removed_exact += 1

        ans = get_tokens_for_segments(data, ok_answers, tokenizer)
        print(ans)

        # label the CA data with only the roundtrip consistent labels
        data_mod = label_CA_data_mod(data, ok_answers)
        CA_data_mod.append(data_mod)
        y_labels += data_mod['true_labels']
        y_preds += data_mod['predicted_labels']
        item_stats = evaluate_model_answer_spans(data_mod['true_labels'], data_mod['predicted_labels'], word_ids)
        answer_stats['FP'] += item_stats['FP']
        answer_stats['TP'] += item_stats['TP']
        answer_stats['FN'] += item_stats['FN']
        answer_stats['jaccard'] += item_stats['jaccard']
        answer_stats['overlap'] += item_stats['overlap']
        

    print('number of predicted: ', num_predicted_answers)
    print('number of removed: ', num_removed)

    confusion_matrix_tokens(y_labels, y_preds, 'TEST')
    pre = answer_stats['TP']/(answer_stats['TP']+answer_stats['FP'])
    rec = answer_stats['TP']/(answer_stats['TP']+answer_stats['FN'])
    f1 = 2 * (pre * rec)/(pre + rec)
    print('Precision, answers: {:.2f}'.format(pre))
    print('Recall, answers: {:.2f}'.format(rec))
    print('Mean Jaccard score: {:.2f}'.format(np.mean(np.ravel(answer_stats['jaccard']))))
    print('Mean answer length diff (predicted - true): {:.2f}'.format(np.mean(np.ravel(answer_stats['overlap']))))

    return CA_data_mod

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
    CA_data_mod = compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map)
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