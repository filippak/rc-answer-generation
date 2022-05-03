import pandas as pd
import numpy as np
import argparse
import random
import copy

labels = [
    "Not relevant",
    "Relevant",
]

BGN = "[BGN]"
END = "[END]"

def get_tokens_and_labels(sentences, ranked_sentence_ids):
    all_context_texts = []
    all_labels = []
    all_marked_sentences = []

    # For each sentence, highlight one sentence in context text
    # label whole text as 0 / 1 
    for idx, sent in enumerate(sentences):
        if idx in ranked_sentence_ids:
            label = 1
        else:
            label = 0
        context_text = []
        for idx2, sent2 in enumerate(sentences):
            if idx2 == idx:
                context_text += [BGN] + sent2 + [END]
                all_marked_sentences.append(sent2)
            else:
                context_text += sent2 # concatenate all sentences to a list of consecutive tokens
        
        all_context_texts.append(context_text)
        all_labels.append(label)

    return all_context_texts, all_labels, all_marked_sentences

def label_data(df):
    relevant_sentence_data = []

    for index, row in df.iterrows():
        sentences = row['context_raw']
        answer = row['correct_answer_raw']
        sent_with_ans_id = row['answer_location']
        relevant_sentence_ids = row['relevant_sentence_ids']
        context_id = row['context_id']
        sent_ids = [sent_with_ans_id]
        
        count = 0
        # add (max 3 including sentence with answer) highest ranked sententces
        for sent_id in relevant_sentence_ids:
            if count < 2 and sent_id != sent_with_ans_id:
                sent_ids.append(sent_id)
                count += 1

        context_text_arr, labels_arr, sentence_arr = get_tokens_and_labels(sentences, sent_ids)
        for c_idx, c_text in enumerate(context_text_arr):
            data_point = {'context_id': int(context_id), 'id': index, 'label': labels_arr[c_idx], 'tokens': c_text, 'answer': answer, 'sentence': sentence_arr[c_idx]}
            relevant_sentence_data.append(data_point)
    
    print(relevant_sentence_data[19])
    print(relevant_sentence_data[20])
    return relevant_sentence_data



def main(args):
    df = pd.read_pickle(args.data_path)
    print('Num data points: ', len(df))

    labeled_data = label_data(df)
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