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
from eval import correct_word_piece_tokens, prediction_output_partial, prediction_output_strict, print_extracted_answers

CRA_TOKENS =  ['[BGN]', '[END]']


def get_model_predictions(data, model):
    """
    Function to get prediction for a data point (datapoint being a tokenized text segment)
    
    Input: data to predict, model to predict with
    Output: predictions for input data (array of tensors on size 1)
    """
    output = model(torch.tensor([data['input_ids']]), attention_mask=torch.tensor([data['attention_mask']]), token_type_ids=torch.tensor([data['token_type_ids']]))
    m = nn.Softmax(dim=2)
    max = m(output.logits)
    out = torch.argmax(max, dim=2)
    return out[0]

def get_CRA_predictions(model, tokenizer, data):
    for i in range(len(data)):
        out = get_model_predictions(data[i], model)
        word_ids = word_ids = data[i].word_ids()
        tokens = tokenizer.convert_ids_to_tokens(data[i]["input_ids"]) # to use if printing results..
        dec = tokenizer.decode(data[i]["input_ids"])
        print('input: ', dec)

        output_labels, output_token_list, output_token_labels, token_word_ids = correct_word_piece_tokens(data[i], out, tokens)
        # get the predictions
        output_token_labels = prediction_output_strict(output_token_labels)
        
        # set the output labels to the data point
        data[i]['predicted_token_labels'] = output_token_labels
        data[i]['token_list'] = output_token_list
        data[i]['token_word_ids'] = token_word_ids
        print_extracted_answers(output_labels, tokens, word_ids)
    return data

def main(args):
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)

    # cheat to get the model in to torch model form
    torch.save(model, args.model_path+'.pkl')
    model = torch.load(args.model_path+'.pkl')
    model.eval()

    with open(args.data_path, "rb") as input_file:
        data = pickle.load(input_file)
    
    prediction_data = get_CRA_predictions(model, tokenizer, data)

    # save the outputs for the validation data. to use for comparison between CA and CRA model outputs
    with open(args.output_path, "wb") as output_file:
        pickle.dump(prediction_data, output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('model_path', type=str, 
        help='path model file', action='store')
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file', action='store')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)