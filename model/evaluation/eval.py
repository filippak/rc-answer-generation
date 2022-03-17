from tkinter import N
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
from transformers import AutoTokenizer
import torch
import torch.nn as nn

CRA_TOKENS =  ['[BGN]', '[END]']

# function that strictly extracts correctly formatted answers
def get_token_segments(labels):
    labels_stats = []
    for idx, label in enumerate(labels):
        if label == 1:
            count = 1
            while idx+count < len(labels):
                next_label = labels[idx+count]
                if next_label != 2:
                    break
                count += 1
            labels_stats.append((idx, count))
    return labels_stats

def correct_word_piece_tokens(tokenized_inputs, labels_in):
    # print('labels: ', labels_in)
    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    previous_word_label = None
    label_ids = []
    for idx, word_idx in enumerate(word_ids):  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(0) # label CLS, SEP as 0..
            previous_word_label = 0
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(labels_in[idx])
            previous_word_label = labels_in[idx]
        else:
            # this token belongs to the same word as the previous 
            # -> must have label 2 if prev is 1 or 2, and 0 otherwise
            if previous_word_label > 0:
                label_ids.append(2)
                previous_word_label = 2
            else:
                label_ids.append(0)
                previous_word_label = 0
        previous_word_idx = word_idx
    
    # print('labels out: ', label_ids)
    return label_ids

# function that modifies the predicted answer outputs so that sequences that starts with a 2, becomes a 1
def prediction_output_modified(output):
    corrected_output = []
    prev_label = 0
    for idx, label in enumerate(output):
        if label == 2 and prev_label == 0:
            corrected_output.append(1)
            prev_label = 1
        else:
            corrected_output.append(label)
            prev_label = label
    return corrected_output

# remove all segments that start with 2
def prediction_output_strict(output):
    corrected_output = []
    prev_label = 0
    for idx, label in enumerate(output):
        if label == 2 and prev_label == 0:
            corrected_output.append(0)
            prev_label = 0
        else:
            corrected_output.append(label)
            prev_label = label
    return corrected_output



def evaluate_model_tokens(labels, predicted):
    FP = 0
    TP = 0
    TN = 0
    FN = 0
    for idx, label in enumerate(labels):
        if label > 0 and predicted[idx] > 0:
            TP += 1
        elif label > 0:
            FP += 1
        elif predicted[idx] > 0:
            FN += 1
    return [FP, TP, FN]

def get_correct_answer_dict(label_segments):
    label_dict = {}
    for a in label_segments:
        key = str(a[0]) + ' ' + str(a[1])
        label_dict[key] = None
    return label_dict

def get_jaccard_score(a, s):
    a_start = a[0]
    a_end = a[0]+a[1]-1
    start = s[0]
    end = s[0]+s[1]-1
    pred_ids = set(range(start, end+1))
    label_ids = set(range(a_start, a_end+1))
    jacc = len(pred_ids.intersection(label_ids)) / len(pred_ids.union(label_ids))
    return jacc


def evaluate_model_answer_spans(true_labels, output_labels):
    FP = 0
    TP = 0
    TN = 0
    FN = 0
    jaccard_scores = []
    segment_lengths = []
    label_segments = get_token_segments(true_labels)
    predicted_segments = get_token_segments(output_labels)
    num_answers = len(label_segments)

    label_dict = get_correct_answer_dict(label_segments)
    output_dict = get_correct_answer_dict(predicted_segments)
    for a in label_segments:
        a_start = a[0]
        a_end = a[0]+a[1]-1
        label_dict_key = str(a[0]) + ' ' + str(a[1])
        for s in predicted_segments:
            start = s[0]
            end = s[0]+s[1]-1
            output_dict_key = str(s[0]) + ' ' + str(s[1])
            # check if there is overlap
            if start <= a_end and end >= a_start:
                jacc = get_jaccard_score(a, s)
                if label_dict[label_dict_key] != None:
                    # there already exists a predicted answer for this segment..
                    segments = label_dict[label_dict_key]['segments']
                    segments.append({'segment':s, 'jaccard': jacc})
                    label_dict[label_dict_key]['segments'] = segments
                    if jacc > label_dict[label_dict_key]['max_jacc']:
                        # update max jaccard score and segment length difference if current answer has higher jaccard score.
                        label_dict[label_dict_key]['max_jacc'] = jacc
                        label_dict[label_dict_key]['segment_length'] = s[1]-a[1]
                else:
                    TP += 1 # only count 1 true positive
                    label_dict[label_dict_key] = {'max_jacc': jacc, 'segments': [{'segment':s, 'jaccard': jacc}], 'segment_length': s[1]-a[1]}
                # break # shound only consider overlap with one answer at a time??
            
            # only count each of the predicted segments as a false negative once (else they are counted once for each correct answer)
            elif output_dict[output_dict_key] == None:
                FN += 1
                output_dict[output_dict_key] = True
        
        # add the max jaccard scores
        if label_dict[label_dict_key] != None:
            jaccard_scores.append(label_dict[label_dict_key]['max_jacc'])
            segment_lengths.append(label_dict[label_dict_key]['segment_length'])


    # the number of correct answers that were missed
    FP += num_answers - TP
    return [FP, TP, FN, jaccard_scores, segment_lengths]

def print_extracted_answers(output_labels, tokens):
    predicted_segments = get_token_segments(output_labels)
    all_l = []
    for segment in predicted_segments:
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
        print(l)
        all_l.append(l)
    return all_l

def evaluate_model(model, tokenizer, data, use_strict):
    FP = 0
    TP = 0
    TN = 0
    FN = 0
    FP_t = 0
    TP_t = 0
    FN_t = 0
    jaccard_scores = []
    segment_lengths = [] # (correct segment length, predicted segment length)
    for i in range(len(data)):
        output = model(torch.tensor([data[i]['input_ids']]), attention_mask=torch.tensor([data[i]['attention_mask']]), token_type_ids=torch.tensor([data[i]['token_type_ids']]), labels=torch.tensor([data[i]['labels']]))
        # print('test idx: ', i)
        # print('instance loss: ', output.loss)
        m = nn.Softmax(dim=2)
        max = m(output.logits)
        out = torch.argmax(max, dim=2)

        tokens = tokenizer.convert_ids_to_tokens(data[i]["input_ids"])
        true_labels = correct_word_piece_tokens(data[i], data[i]['labels']) # replace the -100 labels used in training..
        data[i]['true_labels'] = true_labels
        # print('answers: ')
        # print_extracted_answers(true_labels, tokens)

        # merge the word piece tokens (if word piece token # 1 has label 1, # 2 should have label 2 etc. )
        output_labels = correct_word_piece_tokens(data[i], out[0])
        if use_strict:
            output_labels = prediction_output_strict(output_labels)
        else:
            output_labels = prediction_output_modified(output_labels)
        # set the output labels to the data point
        data[i]['predicted_labels'] = output_labels
        # print('predictions: ')
        # print_extracted_answers(output_labels, tokens)

        a_results = evaluate_model_answer_spans(true_labels, output_labels)
        FP += a_results[0]
        TP += a_results[1]
        FN += a_results[2]
        jaccard_scores += a_results[3]
        segment_lengths += a_results[4]

        t_results = evaluate_model_tokens(true_labels, output_labels)
        FP_t += t_results[0]
        TP_t += t_results[1]
        FN_t += t_results[2]
    
    pre_t = TP_t/(TP_t+FP_t)
    rec_t = TP_t/(TP_t+FN_t)
    f1_t = 2 * (pre_t * rec_t)/(pre_t + rec_t)
    print('Precision, tokens: {:.2f}'.format(pre_t))
    print('Recall, tokens: {:.2f}'.format(rec_t))
    print('F1-score, tokens: {:.2f}'.format(f1_t))
    
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2 * (pre * rec)/(pre + rec)
    print('Precision, answers: {:.2f}'.format(pre))
    print('Recall, answers: {:.2f}'.format(rec))
    print('F1-score, answers: {:.2f}'.format(f1))
    print('Mean Jaccard score: {:.2f}'.format(np.mean(np.ravel(jaccard_scores))))
    print('Mean answer length diff: {:.2f}'.format(np.mean(np.ravel(segment_lengths))))


        

    # calculate precision and recall


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    if args.CRA: # add BGN and END tokens
        num_added_toks = tokenizer.add_tokens(CRA_TOKENS)
        # print('Added ', num_added_toks, 'tokens')

    model = torch.load(args.model_path)
    model.eval()
    with open(args.data_path, "rb") as input_file:
        validation_data = pickle.load(input_file)
    # tokens = tokenizer.convert_ids_to_tokens(validation_data[0]["input_ids"])
    # print(tokens)
    # dec = tokenizer.decode(validation_data[0]["input_ids"])
    # print(dec)
    evaluate_model(model, tokenizer, validation_data, args.strict)

    # save the outputs for the validation data. to use for comparison between CA and CRA model outputs
    with open(args.output_path, "wb") as output_file:
        pickle.dump(validation_data, output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('model_path', type=str, 
        help='path model file', action='store')
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file', action='store')
    parser.add_argument('--strict', dest='strict', action='store_true')
    parser.add_argument('--CRA', dest='CRA', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
