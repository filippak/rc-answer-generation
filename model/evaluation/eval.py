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


def get_token_segments(labels):
    labels_stats = []
    for idx, label in enumerate(labels):
        if label == 1 and idx+1 < len(labels):
            next_label = labels[idx+1]
            count = 2
            while idx+count < len(labels) and next_label in [2, -100]:
                next_label = labels[idx+count]
                count += 1
            labels_stats.append((idx, count-1))
    return labels_stats

def get_all_consecutive_token_segments(labels):
    labels_stats = []
    next_start_idx = -1
    for idx, label in enumerate(labels):
        if idx > next_start_idx and label > 1 and idx+1 < len(labels):
            next_label = labels[idx+1]
            count = 2
            while idx+count < len(labels) and next_label in [2, -100]:
                next_label = labels[idx+count]
                count += 1
            labels_stats.append((idx, count-1))
            next_start_idx = idx+count-1
    return labels_stats

def evaluate_model(model, tokenizer, data):
    FP = 0
    TP = 0
    TN = 0
    FN = 0
    for i in range(len(data)):
        output = model(torch.tensor([data[i]['input_ids']]), attention_mask=torch.tensor([data[i]['attention_mask']]), token_type_ids=torch.tensor([data[i]['token_type_ids']]), labels=torch.tensor([data[i]['labels']]))
        print('test idx: ', i)
        print('instance loss: ', output.loss)
        m = nn.Softmax(dim=2)
        max = m(output.logits)
        out = torch.argmax(max, dim=2)

        tokens = tokenizer.convert_ids_to_tokens(data[i]["input_ids"])
        true_labels = data[i]['labels']

        # TODO: pre-process the output (change 2 -> 1, merge output, add sentences etc.)

        label_segments = get_token_segments(true_labels)
        predicted_segments = get_all_consecutive_token_segments(out[0])
        print('label segments: ', label_segments)
        print('predicted segments: ', predicted_segments)
        num_answers = len(label_segments)
        num_predicted = 0
        for s in predicted_segments:
            start = s[0]
            end = s[0]+s[1]-1
            # compare each prediction to each of the answers
            for a in label_segments:
                a_start = a[0]
                a_end = a[0]+a[1]-1
                # check if there is overlap
                if start <= a_end and end >= a_start:
                    print('segments overlap!')
                    num_predicted += 1
                    pred_ids = set(range(start, end+1))
                    print('pred ids: ',pred_ids)
                    label_ids = set(range(a_start, a_end+1))
                    print('label ids: ',label_ids)
                    jacc = len(pred_ids.intersection(label_ids)) / len(pred_ids.union(label_ids))
                    print('jaccard score: ', jacc)
                    TP += jacc
                    break # shound only consider overlap with one answer at a time??
                    
                else:
                    FN += 1
        # the number of correct answers that were missed
        FP += num_answers - num_predicted 
    print('Precision: ', TP/(TP+FP))
    print('Recall: ', TP/(TP+FN))

        

    # calculate precision and recall


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = torch.load(args.model_path)
    model.eval()
    with open(args.data_path, "rb") as input_file:
        validation_data = pickle.load(input_file)
    # tokens = tokenizer.convert_ids_to_tokens(validation_data[0]["input_ids"])
    # print(tokens)
    # dec = tokenizer.decode(validation_data[0]["input_ids"])
    # print(dec)
    evaluate_model(model, tokenizer, validation_data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('model_path', type=str, 
        help='path model file', action='store')
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
