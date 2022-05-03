from multiprocessing import context
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, ElectraTokenizer
import pickle
import argparse
import random
import copy
from train_CAR_sent_class import load_data


def load_data(path):
    _path = path + '.pkl'
    path_with_id = path + '_with_id.pkl'
    with open(_path, "rb") as input_file:
        data = pickle.load(input_file)
    with open(path_with_id, "rb") as input_file:
        data_id = pickle.load(input_file)
    return data, data_id

def balance_dataset(data, data_with_id):
    neg_data_dict = {}
    balanced_data_with_id = []
    balanced_data = []
    for i in range(len(data)):
        context_id = data_with_id[i]['context_id']
        if context_id not in neg_data_dict:
            neg_data_dict[context_id] = {'train': [], 'with_id': [], 'pos': []}
        # print('context_id: ', context_id)
        if data[i]['label'] == 1:
            balanced_data.append(data[i])
            balanced_data_with_id.append(data_with_id[i])
            neg_data_dict[context_id]['pos'].append(data_with_id[i])
        else:
            neg_data_dict[context_id]['train'].append(data[i])
            neg_data_dict[context_id]['with_id'].append(data_with_id[i])
    
    print('number of positive: ', len(balanced_data))
    for key, value in neg_data_dict.items():
        data_length = len(value['train'])
        idx = np.random.rand(len(value['pos'])) * data_length
        loc = np.floor(idx)
        for v in loc:
            balanced_data.append(value['train'][int(v)])
            balanced_data_with_id.append(value['with_id'][int(v)])
    print('total data points: ', len(balanced_data))

    return balanced_data, balanced_data_with_id





def main(args):
    data, data_with_id = load_data(args.data_path)
    print('data length',len(data))
    print('data with id length',len(data_with_id))
    data_b, data_id_b = balance_dataset(data, data_with_id)

    data_path = args.output_path + '.pkl'
    data_path_id = args.output_path + '_with_id.pkl'
    with open(data_path, "wb") as output_file:
        pickle.dump(data_b, output_file)

    with open(data_path_id, "wb") as output_file:
        pickle.dump(data_id_b, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune bert model for token classification')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to dataframe of pre-parsed data', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)



