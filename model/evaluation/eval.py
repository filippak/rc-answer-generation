from tkinter import N
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

CRA_TOKENS =  ['[BGN]', '[END]']

def get_token_segments(labels, word_ids):
    """
    Function to extract correctly formatted answers
    
    Input: array of labels (can be predicted labels, or true labels)
    Output: array of tuples; (start index of answer, length of answer)
    """
    prev_word_id = None
    labels_stats = []
    for idx, label in enumerate(labels):
        if label == 1 and word_ids[idx] != prev_word_id:
            count = 1
            prev_word_id = word_ids[idx]
            while idx+count < len(labels):
                next_label = labels[idx+count]
                if next_label == 0:
                    break
                if next_label == 1 and word_ids[idx+count] != prev_word_id:
                    break 
                count += 1
            labels_stats.append((idx, count))
    return labels_stats

def correct_word_piece_tokens(tokenized_inputs, labels_in):
    """
    Function to correct labels on given input data.
    This function is used to
    - In the predicted labels, correct the labels of WordPieces that are in the same original token as the previous WordPiece
    - In the true labels, change the -100 labels (used in the case mentioned above, and for CLS, SEP, UNK, .. tokens 
    
    Input: array of data, processed by tokenizer
    Output: array of corrected labels
    """
    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    previous_word_label = None
    label_ids = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            label_ids.append(0) # label CLS, SEP as 0..
            previous_word_label = 0
        elif word_idx != previous_word_idx: 
            if labels_in[idx] <= 0: # [BGN] and [END] can be labeled -100, but should in eval be equal to 0..
                label_ids.append(0)
                previous_word_label = 0
            else:
                label_ids.append(labels_in[idx])
                previous_word_label = labels_in[idx]
        else:
            # this token belongs to the same word as the previous 
            # -> must have label that match the previous label 
            label_ids.append(previous_word_label)
        previous_word_idx = word_idx
    
    return label_ids

def prediction_output_partial(output):
    """
    Function that given the predicted labels, updates the labels to fit intended data output format.
    Specifically, corrects predicted answer spans that start with a 2, to instead start with a 1
    
    Input: array of labels
    Output: array of corrected labels
    """
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

def prediction_output_strict(output):
    """
    Function that given the predicted labels, removes sequences that are incorrectly formatted
    Specifically, removed predicted answer spans that start with a 2
    
    Input: array of labels
    Output: array of corrected labels
    """
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

def confusion_matrix_tokens(labels, predicted, title):
    """
    Function that given labels and predictions computes and plots normalized (by rows) confusion matrix
    """
    # Inspiration from: https://vitalflux.com/python-draw-confusion-matrix-matplotlib/
    c_mat = confusion_matrix(labels, predicted, normalize=None)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(c_mat, cmap=plt.cm.Greens, alpha=0.5)
    for i in range(c_mat.shape[0]):
        for j in range(c_mat.shape[1]):
            val = "{0:.2f}".format(c_mat[i, j])
            ax.text(x=j, y=i,s=val, va='center', ha='center', size='xx-large')
    
    ax.set_xlabel('Predicted label', fontsize=16)    
    ax.xaxis.set_label_position('bottom') 
    plt.ylabel('True label', fontsize=16)
    plt.title(title, fontsize=18)
    # plt.show()
    plt.savefig('./figures/'+title)


def evaluate_model_tokens(labels, predicted):
    """
    Function that extracts stats on token-basis
    
    Input: array of true labels, array of predicted labels
    Output: stats for the two given arrays
    """
    stats = {'FP': 0, 'TP': 0, 'FN': 0}
    for idx, label in enumerate(labels):
        if label == 1 and predicted[idx] == 1:
            stats['TP'] += 1
        if label == 2 and predicted[idx] == 2:
            stats['TP'] += 1
        elif label > 0:
            stats['FN'] += 1
        elif predicted[idx] > 0:
            stats['FP'] += 1
    return stats

def get_correct_answer_dict(label_segments):
    """
    Function that returns dictionary with the input as keys

    Input: array of tuples; (start index of answer, length of answer)
    Output: dict with string versions of tuples as keys
    """
    label_dict = {}
    for a in label_segments:
        key = str(a[0]) + ' ' + str(a[1])
        label_dict[key] = None
    return label_dict

def get_jaccard_score(a, s):
    """
    Function that computes the jaccard score between two sequences

    Input: two tuples; (start index of answer, length of answer). Guaranteed to overlap
    Output: jaccard score computed for the two sequences
    """
    a_start = a[0]
    a_end = a[0]+a[1]-1
    start = s[0]
    end = s[0]+s[1]-1
    pred_ids = set(range(start, end+1))
    label_ids = set(range(a_start, a_end+1))
    jacc = len(pred_ids.intersection(label_ids)) / len(pred_ids.union(label_ids))
    return jacc


def evaluate_model_answer_spans(true_labels, output_labels, word_ids):
    """
    Function that evaluates overlap between true labels and predicted output on an answer level
    An output segment is considered a match if it overlaps with a segment in the true labels
    
    Input: Two arrays; one of true labels and one of predicted labels
    Output: Statistics of comparison between true labels and predictions
    """
    stats = {'FP': 0, 'TP': 0, 'FN': 0, 'jaccard': [], 'overlap': []}
    label_segments = get_token_segments(true_labels, word_ids)
    predicted_segments = get_token_segments(output_labels, word_ids)
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
                    stats['TP'] += 1 # only count 1 true positive
                    label_dict[label_dict_key] = {'max_jacc': jacc, 'segments': [{'segment':s, 'jaccard': jacc}], 'segment_length': s[1]-a[1]}
            
            # only count each of the predicted segments as a false positive once!
            elif output_dict[output_dict_key] == None:
                stats['FP'] += 1
                output_dict[output_dict_key] = True
        
        # add the max jaccard score for the current correct answer (if a match was found)
        if label_dict[label_dict_key] != None:
            stats['jaccard'].append(label_dict[label_dict_key]['max_jacc'])
            stats['overlap'].append(label_dict[label_dict_key]['segment_length'])


    # the number of correct answers that were missed
    stats['FN'] += num_answers - stats['TP']
    return stats

def print_extracted_answers(output_labels, tokens, word_ids):
    """
    Function that prints tokens corresponding to answer segments 
    
    Input: 
    - output_labels: array of labels
    - tokens: array of tokens corresponding to the labels
    Output: array of strings corresponding to the answer segments as present in the labels array
    """
    print('tokens: ',tokens)
    predicted_segments = get_token_segments(output_labels, word_ids)
    all_l = []
    for segment in predicted_segments:
        l = ''
        for i in range(segment[1]):
            token = tokens[segment[0]+i]
            if token.startswith('##'):
                l += token[2:]
            elif len(l) > 0: # not the first word in answer phrase
                l += ' ' + token
            else:
                l += token
        print('answer', l)
        all_l.append(l)
    return all_l


def get_model_predictions(data, model):
    """
    Function to get prediction for a data point (datapoint being a tokenized text segment)
    
    Input: data to predict, model to predict with
    Output: predictions for input data (array of tensors on size 1)
    """
    output = model(torch.tensor([data['input_ids']]), attention_mask=torch.tensor([data['attention_mask']]), token_type_ids=torch.tensor([data['token_type_ids']]), labels=torch.tensor([data['labels']]))
    m = nn.Softmax(dim=2)
    max = m(output.logits)
    out = torch.argmax(max, dim=2)
    return out[0]

def evaluate_model(model, tokenizer, data, use_strict, model_name):
    """
    function to evaluate model on given data
    
    Input: 
    - model to use for prediction
    - tokenizer
    - data: the data to get predictions for
    - use_strict: flag indicating if evaluating strict
    Output: predictions for input data (array of tensors on size 1)
    """
    answer_stats = {'FP': 0, 'TP': 0, 'FN': 0, 'jaccard': [], 'overlap': []}
    token_stats = {'FP': 0, 'TP': 0, 'FN': 0}
    y_labels = []
    y_preds = []
    for i in range(len(data)):
        out = get_model_predictions(data[i], model)
        word_ids = word_ids = data[i].word_ids()
        tokens = tokenizer.convert_ids_to_tokens(data[i]["input_ids"]) # to use if printing results..
        true_labels = correct_word_piece_tokens(data[i], data[i]['labels']) # replace the -100 labels used in training..
        y_labels += true_labels
        data[i]['true_labels'] = true_labels
        # print_extracted_answers(true_labels, tokens, word_ids)

        output_labels = correct_word_piece_tokens(data[i], out)
        if use_strict:
            output_labels = prediction_output_strict(output_labels)
        else:
            output_labels = prediction_output_partial(output_labels)
        
        # set the output labels to the data point
        data[i]['predicted_labels'] = output_labels
        y_preds += output_labels
        # print_extracted_answers(output_labels, tokens, word_ids)

        item_stats = evaluate_model_answer_spans(true_labels, output_labels, word_ids)
        answer_stats['FP'] += item_stats['FP']
        answer_stats['TP'] += item_stats['TP']
        answer_stats['FN'] += item_stats['FN']
        answer_stats['jaccard'] += item_stats['jaccard']
        answer_stats['overlap'] += item_stats['overlap']

        item_token_stats = evaluate_model_tokens(true_labels, output_labels)
        token_stats['FP'] += item_token_stats['FP']
        token_stats['TP'] += item_token_stats['TP']
        token_stats['FN'] += item_token_stats['FN']
    
    # calculate precision and recall, F1-score
    pre_t = token_stats['TP']/(token_stats['TP'] + token_stats['FP'])
    rec_t = token_stats['TP']/(token_stats['TP']+token_stats['FN'])
    f1_t = 2 * (pre_t * rec_t)/(pre_t + rec_t)
    # print('Precision, tokens: {:.2f}'.format(pre_t))
    # print('Recall, tokens: {:.2f}'.format(rec_t))
    # print('F1-score, tokens: {:.2f}'.format(f1_t))
    
    pre = answer_stats['TP']/(answer_stats['TP']+answer_stats['FP'])
    rec = answer_stats['TP']/(answer_stats['TP']+answer_stats['FN'])
    f1 = 2 * (pre * rec)/(pre + rec)
    print('Precision, answers: {:.2f}'.format(pre))
    print('Recall, answers: {:.2f}'.format(rec))
    # print('F1-score, answers: {:.2f}'.format(f1))
    print('Mean Jaccard score: {:.2f}'.format(np.mean(np.ravel(answer_stats['jaccard']))))
    print('Mean answer length diff (predicted - true): {:.2f}'.format(np.mean(np.ravel(answer_stats['overlap']))))
    
    # plot the confusion matrix on token level
    title = ''
    if use_strict:
        title += 'Strict evaluation of '
    else:
        title += 'Partial evaluation of '
    if args.CRA:
        title += 'CR-A '
    else:
        title += 'C-A '
    title += 'model trained with {} weights'.format(model_name)
    confusion_matrix_tokens(y_labels, y_preds, title)



def main(args):
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    if args.CRA: # add BGN and END tokens
        num_added_toks = tokenizer.add_tokens(CRA_TOKENS)

    # cheat to get the model in to torch model form
    torch.save(model, args.model_path+'.pkl')
    model = torch.load(args.model_path+'.pkl')
    model.eval()

    with open(args.data_path, "rb") as input_file:
        validation_data = pickle.load(input_file)

    evaluate_model(model, tokenizer, validation_data, args.strict, args.model_name)

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
    parser.add_argument('model_name', type=str, 
        help='name of model that is being evaluated', action='store')
    parser.add_argument('--strict', dest='strict', action='store_true')
    parser.add_argument('--CRA', dest='CRA', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
