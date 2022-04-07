import stanza
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from parse_answers_and_context import define_stopwords
import argparse

stanza.download('sv', processors='tokenize,pos,lemma,depparse')
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma,depparse')

# Find the word type of the root in every answer in the dataset
def parse_doc(doc, stop_words):
    dict = { 'lemmas': [], 'stop_lemmas': [], 'words': [], 'stop_words': [], 'pos': [], 'deprel': [], 'root': None, 'root_word': None, 'root_pos': None,}
    for sentence in doc.sentences:
        for word in sentence.words:
            w  = re.sub('[^\sa-zåäöA-ZÅÄÖ0-9-]', '', word.text) # don't need this? should already be done in previous script..
            l = word.lemma
            if len(w) > 0:
                dict['lemmas'].append(l.lower())
                dict['words'].append(w.lower())
                dict['pos'].append(word.pos)
                dict['deprel'].append(word.deprel)
                if not word.text in stop_words:
                    dict['stop_lemmas'].append(l.lower())
                    dict['stop_words'].append(w.lower())
                if word.deprel == 'root':
                    dict['root'] = word.lemma
                    dict['root_word'] = word.text
                    dict['root_pos'] = word.pos
    return dict

def add_to_dependency_dict(answer, sent_with_ans, question, ans_doc, sent_doc, q_doc, stop_words):
    ans_dict = parse_doc(ans_doc, stop_words)
    sent_dict = parse_doc(sent_doc, stop_words)
    q_dict = parse_doc(q_doc, stop_words)
    dict_item = { 'answer': answer, 'answer_lemmas': [], 'answer_stop_lemmas': [], 'answer_words': [], 'answer_stop_words': [], 'answer_pos': [], 'answer_deprel': [], 'answer_root': None, 'answer_root_pos': None,
            'sent_with_ans': sent_with_ans, 'sent_lemmas': [],'sent_stop_lemmas': [], 'sent_words': [], 'sent_stop_words': [], 'sent_pos': [], 'sent_deprel': [], 'sent_root': None, 'sent_root_pos': None,
            'question': question, 'q_lemmas': [], 'q_stop_lemmas': [], 'q_words': [], 'q_stop_words': [], 'q_pos': [], 'q_deprel': [], 'q_root': None, 'q_root_pos': None }

    pre_scripts = [('answer_', ans_dict), ('sent_', sent_dict), ('q_', q_dict)]
    for pre in pre_scripts:
        dict_item[pre[0]+'lemmas'] = pre[1]['lemmas']
        dict_item[pre[0]+'stop_lemmas'] = pre[1]['stop_lemmas']
        dict_item[pre[0]+'words'] = pre[1]['words']
        dict_item[pre[0]+'stop_words'] = pre[1]['stop_words']
        dict_item[pre[0]+'pos'] = pre[1]['pos']
        dict_item[pre[0]+'deprel'] = pre[1]['deprel']
        dict_item[pre[0]+'root'] = pre[1]['root']
        dict_item[pre[0]+'root_word'] = pre[1]['root_word']
        dict_item[pre[0]+'root_pos'] = pre[1]['root_pos']
    return dict_item

def dependency_parsing(df, stop_words):
    dependency_dict = []
    for index, row in df.iterrows():
        answer = row['correct_answer_parsed']
        sent_with_ans = row['sent_with_ans']
        question = row['question_parsed']
        ans_doc = nlp(answer)
        sent_doc = nlp(sent_with_ans)
        q_doc = nlp(question)
        dependency_dict.append(add_to_dependency_dict(answer, sent_with_ans, question, ans_doc, sent_doc, q_doc, stop_words))
    
    return dependency_dict

def find_overlap_sent(a_w, q_w):
    overlap = 0
    overlap_w = []
    for w in a_w:
        if w in q_w:
            overlap += 1
            overlap_w.append(w)
    return overlap, overlap_w

# find overlap using the lemmatized versions of the answer sentences and questions
def find_word_overlap(df, remove_stopwords=False, use_lemmas=False):
    overlap_nr = []
    overlap_words = []
    tot_ans_words_arr = []
    tot_q_words_arr = []
    for index, row in df.iterrows():
        q_w = None
        a_w = None
        if remove_stopwords:
            if use_lemmas:
                q_w = row['q_stop_lemmas']
                a_w = row['sent_stop_lemmas']
            else: 
                q_w = row['q_stop_words']
                a_w = row['sent_stop_words']
        else:
            if use_lemmas:
                q_w = row['q_lemmas']
                a_w = row['sent_lemmas']
            else:
                q_w = row['q_words']
                a_w = row['sent_words']
        q_w = list(dict.fromkeys(q_w)) # remove duplicate words
        a_w = list(dict.fromkeys(a_w)) # remove duplicate words
        tot_ans_words_arr.append(len(a_w))
        tot_q_words_arr.append(len(q_w))
        overlap, overlap_w = find_overlap_sent(a_w, q_w)
        
        overlap_words.append(overlap_w)
        overlap_nr.append(overlap)
    
    return overlap_nr, overlap_words, tot_ans_words_arr, tot_q_words_arr

def add_overlap_to_df(df):
    # words with stopwords remaining
    df['word_overlap_count'], df['word_overlap_words'], df['word_ans_wordcount'], \
    df['word_q_wordcount'] = find_word_overlap(df)
    # words stopwords removed
    df['word_stop_overlap_count'], df['word_stop_overlap_words'], df['word_stop_ans_wordcount'], \
    df['word_stop_q_wordcount'] = find_word_overlap(df, True, False)

    # lemmas stopwords remaining
    df['lemma_overlap_count'], df['lemma_overlap_words'], df['lemma_ans_wordcount'], \
    df['lemma_q_wordcount'] = find_word_overlap(df, False, True)
    # lemmas stopwords removed
    df['lemma_stop_overlap_count'], df['lemma_stop_overlap_words'], df['lemma_stop_ans_wordcount'], \
    df['lemma_stop_q_wordcount'] = find_word_overlap(df, True, True)
    return df

def main(args):
    df = pd.read_pickle(args.data_path)
    stop_words = define_stopwords()
    dict = dependency_parsing(df, stop_words)
    dp_dict = pd.DataFrame(dict)

    # fetch the word overlap data
    dp_dict = add_overlap_to_df(dp_dict)


    # save dataframe
    dp_dict = dp_dict.reset_index()
    dp_dict.to_pickle(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')


    args = parser.parse_args()
    main(args)
