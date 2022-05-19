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
from label_using_CA_results import get_tokenized_sentence
import stanza

stanza.download('sv', processors='tokenize')
nlp = stanza.Pipeline(lang='sv', processors='tokenize')

CRA_TOKENS =  ['[BGN]', '[END]']



def get_model_predictions(data, model):
    """
    Function to get prediction for a data point (datapoint being a tokenized text segment)
    
    Input: data to predict, model to predict with
    Output: prediction for input data
    """

    output = model(torch.tensor([data['input_ids']]), attention_mask=torch.tensor([data['attention_mask']]), token_type_ids=torch.tensor([data['token_type_ids']]))
    # print('output: ', output)
    m = nn.Softmax(dim=1)
    max = m(output.logits)
    # print('max: ',max)
    out = torch.argmax(max, dim=1)
    return out[0]


def get_all_model_predictions(model, tokenizer, test_data):
    special_token_ids = tokenizer.convert_tokens_to_ids(CRA_TOKENS + ['[SEP]'])
    sep_token_id = special_token_ids[2]
    bgn_token_id = special_token_ids[0]
    end_token_id = special_token_ids[1]
    relevant_sentence_data = {}
    for i, data in enumerate(test_data):
        if i % 100 == 0:
            print('Processed {} data points'.format(i))
        
        pred = get_model_predictions(data, model)
        # print('pred: ', pred)
        data['predicted_label'] = pred
        if pred == 1:
            # print('sent data: ', relevant_sentence_data)
            dec = tokenizer.decode(data["input_ids"])
            # print(dec)
            # get the current answer
            sep_idx = data["input_ids"].index(sep_token_id)
            context_ids = data["input_ids"][1:sep_idx]
            answer_ids = data["input_ids"][sep_idx+1:-1]
            ans = tokenizer.decode(answer_ids)
            context_id = data['context_id']
            sentence_id = data['sentence_id']
            if context_id in relevant_sentence_data:
                # check if answer is representend
                # add the current answer to the structure and the relevant sentence
                current_context = relevant_sentence_data[context_id]['answers']
                if ans in current_context:
                    current_context[ans].append(sentence_id)
                else: 
                    current_context[ans] = [sentence_id]
            else:
                # save the tokenized context text
                # remove special tokens
                bgn_idx = context_ids.index(bgn_token_id)
                end_idx = context_ids.index(end_token_id)
                tokens = context_ids[:bgn_idx] + context_ids[bgn_idx+1:end_idx] + context_ids[end_idx+1:]

                dec = tokenizer.decode(tokens)
                # print(dec)
                doc = nlp(dec)
                # print(doc)
                text_sent_tokens = []
                for sent in doc.sentences:
                    text_sent_tokens.append(get_tokenized_sentence(sent))
                
                relevant_sentence_data[context_id] = { 'text': text_sent_tokens, 'answers': { ans: [sentence_id] } }
    return relevant_sentence_data
        
def create_CA_data(data_map):
    CRA_data = []
    for context_id, data in data_map.items():
        text = data['text']
        answers = data['answers']
        for answer, sentence_ids in answers.items():
            highlighted_context_text = []
            all_labels = []
            for idx, sent in enumerate(text):
                if idx in sentence_ids:
                    c_sent = sent.copy()
                    c_sent.insert(0, CRA_TOKENS[0])
                    c_sent.append(CRA_TOKENS[1])
                    highlighted_context_text += c_sent
                    labels = np.zeros(len(c_sent))
                else:
                    highlighted_context_text += sent
                    labels = np.zeros(len(sent))
                all_labels.append(labels)
            l = np.concatenate(all_labels).ravel()
            # make context text one array
            data_point = { 'context_id': context_id, 'labels': [], 'tokens': highlighted_context_text, 'answer': answer, 'labels': l }
            # print('data_point: ', data_point)
            CRA_data.append(data_point)
    return CRA_data



def main(args):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)


    num_added_toks = tokenizer.add_tokens(CRA_TOKENS)

    # cheat to get the model in to torch model form
    torch.save(model, args.model_path+'.pkl')
    model = torch.load(args.model_path+'.pkl')
    model.eval()

    with open(args.data_path, "rb") as input_file:
        test_data = pickle.load(input_file)
    print('Number of data points: ',len(test_data))
    # random.shuffle(test_data)
    # test_data = test_data[:20]
    print('Getting predictions..')
    relevant_sentence_data = get_all_model_predictions(model, tokenizer, test_data)
    print('Creating CA data..')
    CRA_data = create_CA_data(relevant_sentence_data)

    # save the CRA_data
    CRA_data_df = pd.DataFrame(CRA_data)
    print('num new data points: ', len(CRA_data_df))

    # save dataframes
    CRA_data_df.to_pickle(args.output_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('model_path', type=str, 
        help='path model file', action='store')
    parser.add_argument('data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to data file', action='store')

    args = parser.parse_args()
    main(args)