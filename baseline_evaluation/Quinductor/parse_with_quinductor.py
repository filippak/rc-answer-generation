# program to run the quinductor
import argparse
import pickle
import pandas as pd
import stanza
import numpy as np
import copy
import matplotlib.pyplot as plt
import udon2
from udon2.helpers import get_deprel_chain
import quinductor as qi

"""
1. Load the existing templates
2. Import the test data
3. dependency parse the sentences of the test data
4. Apply Quinductor to each sentence
5. Extract the generated answer from the Q/A pair
6. Label the data using the same scheme as for the CA / CRA method
7. Evaluate the results using the same evaluation scripts"""

# https://github.com/dkalpakchi/quinductor#using-your-own-templates


stanza.download('sv', processors='tokenize,pos,lemma,depparse')
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)
nlp_ans = stanza.Pipeline(lang='sv', processors='tokenize')

def merge_answer_labels(all_labels):
    num_removed = 0
    num_exact_match = 0
    final_labels = np.zeros(len(all_labels[0]), dtype=int)
    for n_labels in all_labels:
        o_labels = copy.deepcopy(final_labels)
        add_answer_label = True
        for idx, label in enumerate(n_labels):
            if label > 0:
                if o_labels[idx] == 0:
                    o_labels[idx] = label
                elif label != o_labels[idx]:
                    # labels overlap, but are not an exact match
                    # -> don't add the current answer.
                    if add_answer_label: # only count the first for each answer
                        num_removed += 1
                    add_answer_label = False # this means the answers are overlapping, but not equal! -> don't want this
                else:
                    # labels match
                    # -> don't add the current answer.
                    if add_answer_label:
                        num_exact_match += 1
                    add_answer_label = False
        if add_answer_label:
            final_labels = o_labels
    return final_labels, num_removed, num_exact_match

def find_answer_start(answer, sent):
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


def get_all_labels_for_answers(answers, sentence):
    if len(answers) == 0:
        return [np.zeros(len(sentence), dtype=int)], []

    all_labels = []
    all_answers = []
    for ans in answers:
        ans_tok = nlp_ans(ans)
        tokenized_answer = []
        for list in ans_tok.sentences:
            for word in list.words:
                tokenized_answer.append(word.text)
        print('ans tok: ', tokenized_answer)
        idx_s = find_answer_start(tokenized_answer, sentence)
        labels = np.zeros(len(sentence), dtype=int)
        if idx_s != None:
            all_answers += [ans]
            labels[idx_s] = 1
            for i in range(len(tokenized_answer)-1):
                labels[idx_s + i + 1] = 2
        else:
            print('ERROR: could not find start of answer..')
        all_labels.append(labels)
    print('all labels: ', all_labels)
    return all_labels, all_answers

def apply_quinductor(df):
    data = {}
    for index, row in df.iterrows():
        context = row['context_raw'] # array of arrays
        context_id = row['context_id']
        if context_id in data:
            print('context text already processed!')
        else:
            labels_for_context = []
            answers_for_context = []
            total_c_removed = 0
            total_c_match = 0
            for sentence in context:
                print('sentence: ', sentence)
                sent_doc = nlp([sentence])
                # feed this into UDon2 
                trees = udon2.Importer.from_stanza(sent_doc.to_dict())
                # apply the result to Quinductor..
                tools = qi.use('sv', templates_folder='./sv/swequad/1651743828287737')
                res = qi.generate_questions(trees, tools)
                answers = [x.a for x in res]
                # answers_for_context += answers
                # print('answers: ', answers)
                all_labels, answers = get_all_labels_for_answers(answers, sentence)
                answers_for_context += answers
                # iterate over the labels and merge.
                merged_labels, num_removed, num_exact = merge_answer_labels(all_labels)
                print('merged labels: ', merged_labels)
                total_c_removed += num_removed
                total_c_match += num_exact
                labels_for_context += list(merged_labels)
            
            # l = np.concatenate(labels_for_context).ravel()
            data_point = { 'context_id': context_id, 'labels': labels_for_context, 'context': context, 'answers': answers_for_context }
            print('data point: ', data_point)
            print('num removed: ', total_c_removed)
            print('num match: ', total_c_match)
            data[context_id] = data_point
    return list(data.values())

def main(args):
    df = pd.read_pickle(args.data_path)
    print('Num data points: ', len(df))
    qui_labeled_data = apply_quinductor(df)
    qui_labeled_df = pd.DataFrame(qui_labeled_data)
    # save dataframe
    qui_labeled_df.to_pickle(args.output_path)

    # compare labels to those created for the CA dataset


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Quinductor and extract suggested answer phrases')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to data', action='store')
    parser.add_argument('output_path', type=str, 
        help='output data path', action='store')

    args = parser.parse_args()
    main(args)