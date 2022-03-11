import pandas as pd
import numpy as np
import argparse
import random

labels = [
    "0",
    "B-sentence",
    "I-sentence",
]

def get_tokens_and_labels(sentences, ranked_sentence_ids):
    context_text = []
    all_labels = []

    # get the labels and tokens for current sentence
    for idx, sent in enumerate(sentences):
        context_text += sent # concatenate all sentences to a list of consecutive tokens
        labels = np.zeros(len(sent))
        if idx in ranked_sentence_ids:
            # set the labels to 1/2 for the whole sentence!
            labels = np.ones(len(sent)) * 2
            labels[0] = 1
        all_labels.append(labels)
    l = np.concatenate(all_labels).ravel()
    return context_text, l

def label_data(df, df_relevant_sentences):
    relevant_sentence_data = []
    context_to_id = {}
    c_context_id = 0

    for index, row in df.iterrows():
        if not row['context'] in context_to_id:
            context_id = c_context_id
            context_to_id[row['context']] = c_context_id
            c_context_id += 1
        else:
            context_id = context_to_id[row['context']]

        sentences = row['context_raw']
        answer = row['correct_answer_raw']
        sent_with_ans_id = row['answer_location']
        relevant_sentence_ids = df_relevant_sentences.iloc[index]['ranked_matching_sentence_ids']
        sent_ids = [sent_with_ans_id]
        
        count = 0
        # add (max 3 including sentence with answer) highest ranked sententces
        for sent_id in relevant_sentence_ids:
            if count < 2 and sent_id != sent_with_ans_id:
                sent_ids.append(sent_id)
                count += 1

        context_text, labels = get_tokens_and_labels(sentences, sent_ids)
        labels = [ int(x) for x in labels ]
        ans_labels = np.zeros(len(answer), dtype=int)
        data_point = { 'context_id': context_id, 'id': index, 'labels': labels, 'tokens': context_text, 'answer': answer, 'answer_labels': ans_labels }
        relevant_sentence_data.append(data_point)
    
    print('Num context texts: ', c_context_id)
    print(relevant_sentence_data[20])
    return relevant_sentence_data

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

    labeled_data = label_data(df, df_sent)
    labeled_df = pd.DataFrame(labeled_data)

    # split into train and validation set
    df_train, df_val = train_val_split(labeled_df, 0.2)

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