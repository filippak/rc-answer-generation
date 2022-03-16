import pandas as pd
import numpy as np
import argparse
import random

def create_context_to_id_map(df, df_sent):
    context_to_id = {}
    c_context_id = 0
    context_ids = []
    relevant_sentence_ids_arr = []
    for index, row in df.iterrows():
        # add the relevant sentences to the main df
        relevant_sentence_ids = df_sent.iloc[index]['ranked_matching_sentence_ids']
        relevant_sentence_ids_arr.append(relevant_sentence_ids)
        # map the ids
        if not row['context'] in context_to_id:
            context_id = c_context_id
            context_to_id[row['context']] = c_context_id
            c_context_id += 1
        else:
            context_id = context_to_id[row['context']]
        context_ids.append(context_id)
    print('Num context texts: ', len(context_to_id.keys()))
    return context_ids, relevant_sentence_ids_arr


def train_val_split(df, frac):
    train_context_ids = []
    val_context_ids = []
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    num_in_val = int(np.floor(len(df) * frac))
    print('num in validation set: ', num_in_val)
    num = 0
    while num < num_in_val:
        for index, row in df.iterrows():
            
            context_id = row['context_id']
            n = random.random()
            if context_id in train_context_ids:
                df_train = df_train.append(row, ignore_index=True)
            elif context_id in val_context_ids:
                df_val = df_val.append(row, ignore_index=True)
                num += 1
            elif n < frac:
                df_val = df_val.append(row, ignore_index=True)
                val_context_ids.append(context_id)
                num += 1
            else:
                df_train = df_train.append(row, ignore_index=True)
                train_context_ids.append(context_id)
            if num == num_in_val:
                break
    return df_train, df_val



def main(args):
    df = pd.read_pickle(args.data_path)
    df_sent = pd.read_pickle(args.sent_data_path)
    print('Num data points: ', len(df))

    df['context_id'], df['relevant_sentence_ids'] = create_context_to_id_map(df, df_sent)

    # split into train and validation set
    df_train, df_val = train_val_split(df, 0.2)
    print(len(df_train))
    print(len(df_val))

    # save dataframes
    df_train.to_pickle(args.output_path+'_train.pkl')
    df_val.to_pickle(args.output_path+'_eval.pkl')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('sent_data_path', type=str, 
        help='path to sentence data file', action='store')
    parser.add_argument('output_path', type=str,
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)


    args = parser.parse_args()
    main(args)