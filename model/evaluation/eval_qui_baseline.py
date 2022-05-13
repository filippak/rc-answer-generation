import pandas as pd
import numpy as np
import pickle
import argparse
from eval import confusion_matrix_tokens, evaluate_model_answer_spans, evaluate_model_tokens


def main(args):
    subset_ids = [1,2,3,6,11,13,17,20,26,40]
    answer_stats = {'FP': 0, 'TP': 0, 'FN': 0, 'jaccard': [], 'overlap': []}
    token_stats = {'FP': 0, 'TP': 0, 'FN': 0}
    qui_df = pd.read_pickle(args.qui_data_path)
    print('Num qui data points: ', len(qui_df))
    labels_df = pd.read_pickle(args.labels_data_path)
    print('Num data points: ', len(labels_df))
    context_ids = qui_df['context_id'].tolist()
    
    if args.subset:
        context_ids = subset_ids

    all_labels = []
    all_preds = []
    for id in context_ids:
        print('context id: ', id)
        qui_row = qui_df.loc[qui_df['context_id'] == id]
        label_row = labels_df.loc[labels_df['context_id'] == id]
        preds = qui_row['labels'].tolist()[0]
        labels = label_row['labels'].tolist()[0]
        print('token length: ', len(preds))
        all_labels += list(labels)
        all_preds += list(preds)

        # GET THE TOKEN STATS
        item_token_stats = evaluate_model_tokens(labels, preds)
        token_stats['FP'] += item_token_stats['FP']
        token_stats['TP'] += item_token_stats['TP']
        token_stats['FN'] += item_token_stats['FN']

        fake_word_ids = list(range(0, len(labels)))
        item_stats = evaluate_model_answer_spans(labels, preds, fake_word_ids)
        answer_stats['FP'] += item_stats['FP']
        answer_stats['TP'] += item_stats['TP']
        answer_stats['FN'] += item_stats['FN']
        answer_stats['jaccard'] += item_stats['jaccard']
        answer_stats['overlap'] += item_stats['overlap']

    
    pre_t = token_stats['TP']/(token_stats['TP'] + token_stats['FP'])
    rec_t = token_stats['TP']/(token_stats['TP']+token_stats['FN'])
    print('Precision, tokens: {:.2f}'.format(pre_t))
    print('Recall, tokens: {:.2f}'.format(rec_t))

    pre = answer_stats['TP']/(answer_stats['TP']+answer_stats['FP'])
    rec = answer_stats['TP']/(answer_stats['TP']+answer_stats['FN'])
    print('Precision, answers: {:.2f}'.format(pre))
    print('Recall, answers: {:.2f}'.format(rec))
    print('Mean Jaccard score: {:.2f}'.format(np.mean(np.ravel(answer_stats['jaccard']))))
    print('Mean answer length diff (predicted - true): {:.2f}'.format(np.mean(np.ravel(answer_stats['overlap']))))
    
    confusion_matrix_tokens(all_labels, all_preds, 'Quinductor baseline on subset')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('qui_data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('labels_data_path', type=str, 
        help='path to data file', action='store')
    parser.add_argument('--subset', dest='subset', action='store_true')

    args = parser.parse_args()
    main(args)
