from tkinter import N
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import random
from transformers import AutoTokenizer, ElectraTokenizer, AutoModel, AutoModelForSequenceClassification, ElectraForSequenceClassification
import torch
import torch.nn as nn
from eval import confusion_matrix_tokens

CRA_TOKENS =  ['[BGN]', '[END]']



def get_model_predictions(data, model, tokenizer):
    """
    Function to get prediction for a data point (datapoint being a tokenized text segment)
    
    Input: data to predict, model to predict with
    Output: prediction for input data
    """
    # print('data: ', data)
    # decoded_sequence = tokenizer.decode(data['input_ids'])
    # print(decoded_sequence)
    output = model(torch.tensor([data['input_ids']]), attention_mask=torch.tensor([data['attention_mask']]), token_type_ids=torch.tensor([data['token_type_ids']]), labels=torch.tensor([data['label']]))
    # print('output: ', output)
    m = nn.Softmax(dim=1)
    max = m(output.logits)
    # print('max: ',max)
    out = torch.argmax(max, dim=1)
    # print('prediction: ', out[0])
    # print('label: ', data['label'])
    return out[0]

def evaluate_model(model, tokenizer, data, model_name):
    """
    function to evaluate model on given data
    """
    stats = {'FP': 1e-6, 'TP': 1e-6, 'FN': 1e-6}
    y_labels = []
    y_preds = []
    for i in range(len(data)):
        out = get_model_predictions(data[i], model, tokenizer)
        y_preds.append(out)
        y_labels.append(data[i]['label'])

        tokens = tokenizer.convert_ids_to_tokens(data[i]["input_ids"]) # to use if printing results..
        # print_extracted_answers(true_labels, tokens)

        if data[i]['label'] == 1 and out == 1:
            stats['TP'] += 1
        elif data[i]['label'] > 0:
            stats['FN'] += 1
        elif out > 0:
            stats['FP'] += 1
        
    
    # calculate precision and recall, F1-score
    pre_t = stats['TP']/(stats['TP'] + stats['FP'])
    rec_t = stats['TP']/(stats['TP']+stats['FN'])
    # f1_t = 2 * (pre_t * rec_t)/(pre_t + rec_t)
    print('Precision: {:.2f}'.format(pre_t))
    print('Recall: {:.2f}'.format(rec_t))
    # print('F1-score: {:.2f}'.format(f1_t))
    
    
    # plot the confusion matrix on token level
    title = 'CA-R model trained on {} labeled data'.format(model_name)
    confusion_matrix_tokens(y_labels, y_preds, title)



def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    # tokenizer = ElectraTokenizer.from_pretrained('KB/electra-base-swedish-cased-discriminator')
    # model = ElectraForSequenceClassification.from_pretrained(args.model_path, num_labels=2)

    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)

    # cheat to get the model in to torch model form
    torch.save(model, args.model_path+'.pkl')
    model = torch.load(args.model_path+'.pkl')
    model.eval()

    with open(args.data_path, "rb") as input_file:
        validation_data = pickle.load(input_file)
    # random.shuffle(validation_data)
    # validation_data = validation_data[:20]
    evaluate_model(model, tokenizer, validation_data, args.model_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('model_path', type=str, 
        help='path model file', action='store')
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('model_name', type=str, 
        help='name of model that is being evaluated', action='store')

    args = parser.parse_args()
    main(args)
