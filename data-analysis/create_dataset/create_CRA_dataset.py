import pandas as pd
import numpy as np
import argparse
from create_CA_dataset import find_answer_start
from create_CAR_dataset import train_val_split


labels = [
    "0",
    "B-sentence",
    "I-sentence",
]
BGN = "[BGN]"
END = "[END]"

def get_tokens_and_labels(sentences, ranked_sentence_ids, answer):
    context_text = []
    all_labels = []
    sent_match_id = ranked_sentence_ids[0]

    # get the labels and tokens for current sentence
    for idx, sent in enumerate(sentences):
        if idx in ranked_sentence_ids:
            c_sent = sent.copy()
            c_sent.insert(0, BGN)
            c_sent.append(END)
            context_text += c_sent
            labels = np.zeros(len(c_sent))

            if idx == sent_match_id:
                idx_s = find_answer_start(answer, c_sent)
                if idx_s != None:
                    labels[idx_s] = 1
                    for i in range(len(answer)-1):
                        labels[idx_s + i + 1] = 2
                else:
                    print('ERROR: could not find start of answer..')
                    print('ans: ', answer)
                    print('sentence: ', sent)

        else:
            context_text += sent # concatenate all sentences to a list of consecutive tokens
            labels = np.zeros(len(sent))

        all_labels.append(labels)
    l = np.concatenate(all_labels).ravel()
    return context_text, l

# create the dataset with the corresponding labels
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

        context_text, labels = get_tokens_and_labels(sentences, sent_ids, answer)
        labels = [ int(x) for x in labels ]
        data_point = { 'context_id': context_id, 'id': index, 'labels': labels, 'tokens': context_text, 'answer': answer }
        relevant_sentence_data.append(data_point)
    
    
    print('Num context texts: ', c_context_id)
    print(relevant_sentence_data[4])
    return relevant_sentence_data

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


    args = parser.parse_args()
    main(args)