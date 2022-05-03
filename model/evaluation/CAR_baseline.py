import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
from transformers import AutoTokenizer
from eval import confusion_matrix_tokens

CRA_TOKENS =  ['[BGN]', '[END]']

def get_baseline_prediction(tokenizer, data):
    # print(data)
    decoded_sequence = tokenizer.decode(data['input_ids'])
    # print(decoded_sequence)
    splt = decoded_sequence.split('[SEP]')
    context = splt[0]
    ans = splt[1]
    # check if the sentence that is marked, contains the answer
    start = context.split('[END]')
    sent = start[0].split('[BGN]')

    if ans in sent[1]:
        return 1
    return 0


def evaluate_model(tokenizer, data):
    """
    function to evaluate model on given data
    """
    stats = {'FP': 1e-6, 'TP': 1e-6, 'FN': 1e-6}
    y_labels = []
    y_preds = []
    for i in range(len(data)):
        pred = get_baseline_prediction(tokenizer, data[i])
        y_preds.append(pred)
        y_labels.append(data[i]['label'])


        if data[i]['label'] == 1 and pred == 1:
            stats['TP'] += 1
        elif data[i]['label'] > 0:
            stats['FN'] += 1
        elif pred > 0:
            stats['FP'] += 1
        
    
    # calculate precision and recall, F1-score
    pre_t = stats['TP']/(stats['TP'] + stats['FP'])
    rec_t = stats['TP']/(stats['TP']+stats['FN'])
    # f1_t = 2 * (pre_t * rec_t)/(pre_t + rec_t)
    print('Precision: {:.2f}'.format(pre_t))
    print('Recall: {:.2f}'.format(rec_t))
    # print('F1-score: {:.2f}'.format(f1_t))
    
    
    # plot the confusion matrix on token level
    title = 'CA-R model baseline'
    confusion_matrix_tokens(y_labels, y_preds, title)



def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)


    with open(args.data_path, "rb") as input_file:
        validation_data = pickle.load(input_file)
    # random.shuffle(validation_data)
    # validation_data = validation_data[:100]
    evaluate_model(tokenizer, validation_data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')

    args = parser.parse_args()
    main(args)