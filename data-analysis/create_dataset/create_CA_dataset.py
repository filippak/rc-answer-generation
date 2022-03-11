import pandas as pd
import numpy as np
import argparse

labels = [
    "0",
    "B-answer",
    "I-answer",
]

def find_answer_start(answer, sent):
    answer = [x.lower() for x in answer]
    sent = [x.lower() for x in sent]
    for idx, word in enumerate(sent):
        if answer[0] in word:
            is_match = True
            if len(answer) > 1:
                for i in range(1, len(answer)):
                    c_len = idx+i
                    if c_len < len(sent):
                        c_ans = answer[i]
                        c_word = sent[idx+i]
                        if c_ans not in c_word:
                            is_match = False
            if is_match:
                return idx
    return None

def get_tokens_and_labels(sentences, answer, sent_with_ans_id):
    context_text = []
    all_labels = []

    # get the labels and tokens for current sentence
    for idx, sent in enumerate(sentences):
        context_text += sent # concatenate all sentences to a list of consecutive tokens
        labels = np.zeros(len(sent))
        if idx == sent_with_ans_id:
            # the answer is contained in this sentence!
            idx_s = find_answer_start(answer, sent)
            if idx_s != None:
                labels[idx_s] = 1
                for i in range(len(answer)-1):
                    labels[idx_s + i + 1] = 2
            else:
                print('ERROR: could not find start of answer..')
                print('ans: ', answer)
                print('sentence: ', sent)
        all_labels.append(labels)
    l = np.concatenate(all_labels).ravel()
    return context_text, l

def label_data(df):
    data_map = {}
    num_removed = 0
    for index, row in df.iterrows():
        sentences = row['context_raw']
        sent_with_ans_id = row['answer_location']
        answer = row['correct_answer_raw']

        context_text, labels = get_tokens_and_labels(sentences, answer, sent_with_ans_id)

        # check if the current text is in the data map, and update the labels accordingly!
        if row['context'] in data_map:
            old_point = data_map[row['context']]
            o_labels = old_point['labels'].copy()
            add_answer_label = True
            for idx, label in enumerate(labels):
                if label > 0:
                    if o_labels[idx] == 0:
                        o_labels[idx] = label
                    elif label != o_labels[idx]:
                        # labels overlap, but are not an exact match
                        # -> don't add the current answer.
                        add_answer_label = False # this means the answers are overlapping, but not equal! -> don't want this
                        num_removed += 1
                    else:
                        # labels match
                        # -> don't add the current answer.
                        add_answer_label = False
                        num_removed += 1
            if add_answer_label:
                old_point['labels'] = o_labels
                old_point['answers'].append(answer)
                data_map[row['context']] = old_point
                
        else:
            data_point = { 'id': index, 'labels': labels, 'tokens': context_text, 'answers': [answer] }
            data_map[row['context']] = data_point
    
    print('number of overlapping answers (removed): ', num_removed)

    for v in data_map.values():
        v['labels'] = [ int(x) for x in v['labels']]
    labeled_data = list(data_map.values())
    print('num labeled data points: ', len(labeled_data))
    return labeled_data

def main(args):
    df = pd.read_pickle(args.data_path)
    print('Num data points: ', len(df))

    labeled_data = label_data(df)
    labeled_df = pd.DataFrame(labeled_data)

    # save dataframe
    labeled_df.to_pickle(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('output_path', type=str,
        help='path to output file where the parsed data will be stored', action='store')


    args = parser.parse_args()
    main(args)