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
    tokens = data['token_list']
    all_l = []
    for segment in segments:
        l = ''
        for i in range(segment[1]):
            token = tokens[segment[0]+i]
            if len(l) > 0: # not the first word in answer phrase
                l += ' ' + token
            else:
                l += token
        all_l.append(l)
    return all_l

def label_CA_data_mod(data, ok_answers):
    """
    Function to label the CA data with only the rountrip consistent answers
    """
    new_data = copy.deepcopy(data)
    new_predictions = [0 for _ in range(len(data['predicted_token_labels']))]
    for ans in ok_answers:
        for i in range(ans[1]):
            if i == 0:
                new_predictions[ans[0]+i] = 1
            else:
                new_predictions[ans[0]+i] = 2
    new_data['predicted_token_labels'] = new_predictions
    return new_data

def get_best_segments(segments, CRA_segments):
    ok_answers = []
    for s in segments:
        start = s[0]
        end = s[0]+s[1]-1
        added = False
        # if the answer overlaps with any of the CRA answers, it is added to the final answers
        for s_cra in CRA_segments:
            start_cra = s_cra[0]
            end_cra = s_cra[0]+s_cra[1]-1
            if not added and start <= end_cra and end >= start_cra:
                #there is overlap!
                # jacc = get_jaccard_score(s, s_cra)
                # if jacc > 0.5:
                ok_answers.append(s)
                added = True
    return ok_answers

def save_extracted_answers(filename, data):
    with open(filename, 'w') as out:
        for id, answers in data.items():
            out.write('-------------------'+ '\n')
            out.write('Context id: {}'.format(id) + '\n')
            out.write('Texter\n')
            for a in answers:
                out.write(a + '\n')


def compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map):
    """ data[i]['token_list'] = true_token_list
    data[i]['token_word_ids'] = token_word_ids
     data[i]['true_token_labels'] = true_token_labels
     'predicted_token_labels' """
    num_removed = 0
    num_predicted_answers = 0
    CA_data_mod = []
    y_labels = []
    y_preds  = []
    answer_stats = {'FP': 0, 'TP': 0, 'FN': 0, 'jaccard': [], 'overlap': [], 'pred_length':[]}
    answer_phrases = {}
    CA_answer_phrases = {}
    for data in CA_data:
        # word_ids = data.word_ids()
        segments = get_token_segments(data['predicted_token_labels'], data['token_word_ids'])
        context_id = data['context_id']
        print('context text id: ', context_id)
        dec = tokenizer.decode(data["input_ids"][1:-1])
        print('text: ', dec)
        CA_answers = get_tokens_for_segments(data, segments, tokenizer)
        CA_answer_phrases[context_id] = CA_answers
        CRA_data_context = context_text_map[context_id]
        CRA_segments = []
        # collect all CRA segments!
        for CRA_data in CRA_data_context:
            CRA_segments += get_token_segments(CRA_data['predicted_token_labels'], CRA_data['token_word_ids'])
        
        ok_answers = get_best_segments(segments, CRA_segments)
        print('Number of ok answers: ',len(ok_answers))

        ans = get_tokens_for_segments(data, ok_answers, tokenizer)
        # TODO: get the relevant sentences for this answer
        print(ans)
        answer_phrases[context_id] = ans

        # label the CA data with only the roundtrip consistent labels
        data_mod = label_CA_data_mod(data, ok_answers)
        CA_data_mod.append(data_mod)
        y_labels += data_mod['true_token_labels']
        y_preds += data_mod['predicted_token_labels']
        item_stats, scores = evaluate_model_answer_spans(data_mod['true_token_labels'], data_mod['predicted_token_labels'], data_mod['token_word_ids'])
        answer_stats['FP'] += item_stats['FP']
        answer_stats['TP'] += item_stats['TP']
        answer_stats['FN'] += item_stats['FN']
        for val in scores.values():
            if val is not None:
                answer_stats['pred_length'] += [s['pred_length'] for s in val]
                answer_stats['jaccard'] += [s['jaccard'] for s in val]
                answer_stats['overlap'] += [s['overlap'] for s in val]

    save_extracted_answers('./data/roundtrip/subset_10/final_answers.txt', answer_phrases)
    save_extracted_answers('./data/roundtrip/subset_10/CA_final_answers.txt', CA_answer_phrases)
    print('number of predicted: ', num_predicted_answers)
    print('number of removed: ', num_removed)

    title = 'Roundtrip consistent results'
    confusion_matrix_tokens(y_labels, y_preds, title)
    pre = answer_stats['TP']/(answer_stats['TP']+answer_stats['FP'])
    rec = answer_stats['TP']/(answer_stats['TP']+answer_stats['FN'])
    f1 = 2 * (pre * rec)/(pre + rec)
    print('Precision, answers: {:.2f}'.format(pre))
    print('Recall, answers: {:.2f}'.format(rec))
    print('Mean length of the predicted answers: {:.2f}'.format(np.mean(np.ravel(answer_stats['pred_length']))))
    print('Mean Jaccard score: {:.2f}'.format(np.mean(np.ravel(answer_stats['jaccard']))))
    print('Mean answer length diff (predicted - true): {:.2f}'.format(np.mean(np.ravel(answer_stats['overlap']))))

    return CA_data_mod

def compare_CA_CRA_predictions(CA_data, CRA_data, tokenizer):
    # get the predicted labels for each context text
    context_text_map = get_CRA_map(CRA_data)
    print('len CRA map: ', len(context_text_map.keys()))
    CA_data_mod = compare_token_segments(CA_data, CRA_data, tokenizer, context_text_map)

    # TODO: save the CA_data_mod for evaluation!
    



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