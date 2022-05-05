import pandas as pd
import numpy as np
import argparse
import random
import copy
import pickle
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eval import get_token_segments
import stanza

stanza.download('sv', processors='tokenize')
nlp = stanza.Pipeline(lang='sv', processors='tokenize')

labels = [
    "Not relevant",
    "Relevant",
]

BGN = "[BGN]"
END = "[END]"


def get_answers_from_segments(segments, tokens):
    all_l = []
    for segment in segments:
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
        word_ids = data.word_ids()
        tokens = tokenizer.convert_ids_to_tokens(data["input_ids"])
        segments = get_token_segments(data['predicted_labels'], word_ids)
        answers = get_answers_from_segments(segments, tokens)
        # parse the context text into original
        dec = tokenizer.decode(data["input_ids"][1:-1])
        doc = nlp(dec)
        text_sent_tokens = []
        for sent in doc.sentences:
            text_sent_tokens.append(get_tokenized_sentence(sent))
       
        for a in answers:
            num_answers += 1
            ans = nlp(a)
            ans_tokens = get_tokenized_sentence(ans.sentences[0])
            for idx, sentence in enumerate(text_sent_tokens):
                # sentence is now a array of tokens
                context_text = []
                sentence_id = -1
                for idx2, sent2 in enumerate(text_sent_tokens):
                    # print(sent2)
                    if idx2 == idx:
                        context_text += [BGN] + sent2 + [END]
                        sentence_id = idx2
                    else:
                        context_text += sent2 # concatenate all sentences to a list of consecutive tokens
                
                data_point = {'context_id': int(context_id), 'tokens': context_text, 'answer': ans_tokens, 'sentence_id': sentence_id, 'label': -1} # adding the label so that it will work with the same tokenization script
                labeled_data.append(data_point)
    
    # print(labeled_data[18])
    # print(labeled_data[19])
    print('num extracted answers: ', num_answers)
    return labeled_data

def divide_CA_data(CA_data):
    selected_data = []
    id_to_length_map = {}
    for data in CA_data:
        data_len = len(data["input_ids"])
        context_id = data['context_id']
        if not context_id in id_to_length_map:
            id_to_length_map[context_id] = data_len

    lengths, ids = zip(*sorted(zip(list(id_to_length_map.values()), list(id_to_length_map.keys()))))

    indexes = np.linspace(0, len(lengths) - 1, 10, dtype='int')
    selected_context_texts = []
    for idx in indexes:
        selected_context_texts.append(ids[idx])
    print('context text ids: ',selected_context_texts)
    return selected_context_texts

def get_CA_subset_data(CA_data, c_ids):
    CA_subset_data = []
    for data in CA_data:
        context_id = data['context_id']
        if context_id in c_ids:
            CA_subset_data.append(data)
    
    return CA_subset_data


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

    with open(args.data_path, "rb") as input_file:
        CA_data = pickle.load(input_file)
    
    # CA_data is an array of already tokenized data, where each of the answers are marked with labels

    # select a subset of context texts for human evaluation
    if args.subset:
        selected_ids = divide_CA_data(CA_data)
        CA_data = get_CA_subset_data(CA_data, selected_ids)
        with open(args.output_path+'_CA_predictions.pkl', "wb") as output_file:
            pickle.dump(CA_data, output_file)



    labeled_data = label_data(CA_data, tokenizer)
    labeled_df = pd.DataFrame(labeled_data)
    print('num new data points: ', len(labeled_df))

    # save dataframes
    labeled_df.to_pickle(args.output_path+'_CAR.pkl')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to file with dataframe', action='store')
    parser.add_argument('output_path', type=str,
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--subset', dest='subset', action='store_true')

    args = parser.parse_args()
    main(args)