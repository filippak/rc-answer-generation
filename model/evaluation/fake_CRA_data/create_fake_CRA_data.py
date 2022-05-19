import pandas as pd
import numpy as np
import argparse
import random
import copy
import pickle
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import stanza

stanza.download('sv', processors='tokenize')
nlp = stanza.Pipeline(lang='sv', processors='tokenize')


BGN = "[BGN]"
END = "[END]"


def get_tokenized_sentence(arr):
    tokens = []
    for raw_word in arr.words:
        tokens.append(raw_word.text)
    return tokens

def label_data(CA_data, tokenizer):
    num_answers = 0
    labeled_data = []
    for data in CA_data:
        context_id = data['context_id']
        # parse the context text into original
        dec = tokenizer.decode(data["input_ids"][1:-1])
        doc = nlp(dec)
        text_sent_tokens = []
        for sent in doc.sentences:
            text_sent_tokens.append(get_tokenized_sentence(sent))
            
        for idx, sentence in enumerate(text_sent_tokens):
            # sentence is now a array of tokens
            context_text = []
            for idx2, sent2 in enumerate(text_sent_tokens):
                if idx2 == idx:
                    context_text += [BGN] + sent2 + [END]
                else:
                    context_text += sent2 # concatenate all sentences to a list of consecutive tokens
            labels = np.zeros(len(context_text), dtype=int)
            data_point = {'context_id': int(context_id), 'tokens': context_text, 'labels': labels} # adding the label so that it will work with the same tokenization script
            labeled_data.append(data_point)
            
    print(labeled_data[18])
    print(labeled_data[19])
    return labeled_data



def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    with open(args.data_path, "rb") as input_file:
        CA_data = pickle.load(input_file)


    labeled_data = label_data(CA_data, tokenizer)
    labeled_df = pd.DataFrame(labeled_data)
    print('num new data points: ', len(labeled_df))

    # save dataframes
    labeled_df.to_pickle(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to file with dataframe', action='store')
    parser.add_argument('output_path', type=str,
        help='path to output file where the parsed data will be stored', action='store')

    args = parser.parse_args()
    main(args)