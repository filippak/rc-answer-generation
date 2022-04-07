# Load necessary libraries
import pandas as pd
import numpy as np
import argparse


def find_missing_words(overlap_words, q_words):
    missing_words = []
    for q_w in q_words:
        if not q_w in overlap_words:
            missing_words.append(q_w)
    return missing_words
        

def check_overlap_class(overlap_words, q_words, s_words):
    # TODO: check that overlap words are the same as s_words!
    overlap = 0
    missing_words = []
    for word in q_words:
        if word in s_words:
            overlap += 1
        else:
           missing_words.append(word) 
    
    if overlap == len(q_words):
        return 1, []
    else:
        return 3, missing_words

def get_matching_sentences(data):
    matching_sentences = []
    matching_original_sentences = []
    for value in data.values():
       matching_sentences.append(' '.join(value['sentence_words'])) 
       matching_original_sentences.append(' '.join(value['original_sent_words'])) 
    return matching_sentences, matching_original_sentences

# in the df_dp_train the columns q_words and sent_words holds the lemmatized words of the question and answer sentence.
# df_train['context_parsed_tok_lemma'] holds the tokenized, lemmatized contexts
def collect_relevant_sentences(df, df_dp, remove_stopwords=True, use_lemmas=False):
    all_relevant_sentences = []
    all_relevant_sentence_idxs = []
    for index, row in df_dp.iterrows():
        a_loc = df.iloc[index]['answer_location']
        # given parameters, get the column names from which to get the data
        df_dp_col = ''
        df_col = ''
        df_overlap_col = ''
        if remove_stopwords:
            df_dp_col = 'stop_'
            df_col = '_stop'
            df_overlap_col = '_stop'
        if use_lemmas:
            df_col = '_lemma'+df_col
            df_overlap_col = 'lemma'+df_overlap_col
            df_dp_col += 'lemmas'
        else:
            df_dp_col += 'words'
            df_overlap_col = 'word'+df_overlap_col

        q_w = row['q_'+df_dp_col]
        a_w = row['sent_'+df_dp_col]
        c_w = df.iloc[index]['context_parsed_tok'+df_col]
        # get the original sentences for answer and context
        q_original = row['q_words']
        a_original = row['sent_words']
        c_original = df.iloc[index]['context_parsed_tok']

        # remove duplicate words
        q_w = list(dict.fromkeys(q_w))
        a_w = list(dict.fromkeys(a_w))
        c_w = [ list(dict.fromkeys(w)) for w in c_w ]

        relevant_sentences = {}
        overlap_class, missing_words_sent = check_overlap_class(c_w[a_loc], q_w, a_w)
        # always add the sentence where the answer is to the relevant sentences
        relevant_sentences[a_loc] = {'original_sent_words': a_original, 'sentence_words': a_w, 'overlap_words': row[df_overlap_col+'_overlap_words'], 'overlap_count': row[df_overlap_col+'_overlap_count']}

        words_in_context = set()
        for c_idx, c_sent in enumerate(c_w):
            for word in q_w:
                if word in c_sent:
                    words_in_context.add(word)
                    if c_idx != a_loc and c_idx in relevant_sentences:
                        # this sentence has already been detected to include overlapping words!
                        if not word in relevant_sentences[c_idx]['overlap_words']:
                            relevant_sentences[c_idx]['overlap_words'].append(word)
                            relevant_sentences[c_idx]['overlap_count'] += 1
                    elif c_idx != a_loc:
                        relevant_sentences[c_idx] = {'original_sent_words': c_original[c_idx], 'sentence_words': c_sent, 'overlap_words': [word], 'overlap_count': 1}
        
        missing_words_context = []
        if overlap_class == 3:
            overlap_words = list(words_in_context)
            if len(overlap_words) == len(q_w):
                overlap_class = 2
            else:
                missing_words_context = find_missing_words(overlap_words, q_w)
        matching_sentences, matching_original_sentences = get_matching_sentences(relevant_sentences)  
        all_relevant_sentences.append({
            'context_'+df_dp_col: c_w, 'sent_'+df_dp_col: a_w, 'sent_original': a_original, 'q_'+df_dp_col: q_w, 'q_original': q_original, 'overlap_class': overlap_class,
            'num_overlap_sentences': len(relevant_sentences), 'missing_words_sent': missing_words_sent,
            'missing_words_context': missing_words_context, 'matching_sentence_ids': list(relevant_sentences.keys()), 'matching_sentences': matching_sentences, 'matching_original_sentences': matching_original_sentences, 'data': relevant_sentences
            })
        all_relevant_sentence_idxs.append(list(relevant_sentences.keys()))
    return all_relevant_sentences

# rank relevant sentences based on tf-idf
def rank_matching_sentences(df, df_tf_idf):
    ranked_sentences = []
    ranked_original_sentences = []
    ranked_sentence_ids = []
    for index, row in df.iterrows():
        matching_sentence_scores = []
        matching_sentences = row['matching_sentences']
        matching_original_sentences = row['matching_original_sentences']
        matching_sentence_ids = row['matching_sentence_ids']
        q_lemmas = row['q_stop_lemmas']
        tf_idf = df_tf_idf.iloc[index] # Have tried to change this to the other corpus -- output is the same
        for sent in matching_sentences:
            sentence_score = 0
            for w in q_lemmas:
                if w in tf_idf:
                    word_score = tf_idf[w]
                    if w in sent:
                        sentence_score += word_score
                else:
                    print('word is missing in tf-idf table! ', w)
            matching_sentence_scores.append(sentence_score)
        # sort the sentences based on score
        
        matching_sentence_scores, matching_sentences, matching_sentence_ids, matching_original_sentences = zip(*sorted(zip(matching_sentence_scores, matching_sentences, matching_sentence_ids, matching_original_sentences), reverse=True))
        ranked_sentences.append(matching_sentences)
        ranked_sentence_ids.append(matching_sentence_ids)
        ranked_original_sentences.append(matching_original_sentences)
    return ranked_sentences, ranked_sentence_ids, ranked_original_sentences

def rank_sentences_add_to_df(df, df_tf_idf):
    ranked_sentences, ranked_sentence_ids, ranked_original_sentences = rank_matching_sentences(df, df_tf_idf)
    df['ranked_matching_sentences'] = ranked_sentences
    df['ranked_matching_sentence_ids'] = ranked_sentence_ids
    df['ranked_matching_original_sentences'] = ranked_original_sentences
    return df


def main(args):
    df = pd.read_pickle(args.data_path)
    df_dp = pd.read_pickle(args.dp_data_path)
    df_tf_idf_lemma_stop = pd.read_pickle(args.tf_idf_data_path)


    # word overlap 
    relevant_sentences_word = collect_relevant_sentences(df, df_dp, False, False)
    df_word = pd.DataFrame(relevant_sentences_word)
    # df_word = rank_sentences_add_to_df(df_word, df_tf_idf_lemma_stop)

    # word overlap, stopwords
    relevant_sentences_word_stop = collect_relevant_sentences(df, df_dp)
    df_word_stop = pd.DataFrame(relevant_sentences_word_stop)
    # df_word_stop = rank_sentences_add_to_df(df_word_stop, df_tf_idf_lemma_stop)

    # lemma overlap
    all_relevant_sentences_lemma = collect_relevant_sentences(df, df_dp, False, True)
    df_lemma = pd.DataFrame(all_relevant_sentences_lemma)
    # df_lemma = rank_sentences_add_to_df(df_lemma, df_tf_idf_lemma_stop)
    
    # lemma overlap, stopwords
    relevant_sentences_lemma_stop = collect_relevant_sentences(df, df_dp, True, True)
    df_lemma_stop = pd.DataFrame(relevant_sentences_lemma_stop)
    df_lemma_stop = rank_sentences_add_to_df(df_lemma_stop, df_tf_idf_lemma_stop)

        
    # save dataframes
    df_word.to_pickle(args.output_path + 'word.pkl')
    df_word_stop.to_pickle(args.output_path + 'word_stop.pkl')
    df_lemma.to_pickle(args.output_path + 'lemma.pkl')
    df_lemma_stop.to_pickle(args.output_path + 'lemma_stop.pkl')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('dp_data_path', type=str, 
        help='path to second json file', action='store')
    parser.add_argument('tf_idf_data_path', type=str, 
        help='path to tf_idf file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')


    args = parser.parse_args()
    main(args)