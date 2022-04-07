# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import stanza
from nltk.corpus import stopwords
import argparse

stanza.download('sv', processors='tokenize,pos,lemma,depparse')
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma,depparse')

def define_stopwords():
    stop_words = set(stopwords.words('swedish'))
    stop_words.add('vad')
    stop_words.add('vems')
    stop_words.add('varifrån')
    stop_words.add('vemvilka') # one question has vem/vilka as question word, does not get parsed as 2 words..
    # ['vad', 'hur', 'när', 'var', 'varifrån', 'varför', 'vart', 'vilken', 'vilket', 'vilka', 'vem', 'vems'] 
    stopwords_list = list(stop_words)
    stopwords_list.sort()
    print(stopwords_list)
    return stop_words

def make_sentences_from_tokens(doc, stop_words):
    raw_tok_sentences = []
    all_sentences = []
    all_tok_sentences = []
    all_tok_stop_sentences = []
    all_tok_lemma_sentences = []
    all_tok_lemma_stop_sentences = []
    for sentence in doc.sentences:
        current_raw_tok_sentence = []
        current_sentence = []
        current_tok_sentence = []
        current_tok_stop_sentence = []
        current_tok_lemma_sentence = []
        current_tok_lemma_stop_sentence = []
        for word in sentence.words:
            # add the raw tokenized sentence (to be used by BERT)
            current_raw_tok_sentence.append(word.text)
            # only add if character is letter or number (removes , . ? ! etc.)
            w  = re.sub('[^\sa-zåäöA-ZÅÄÖ0-9-]', '', word.text)
            l = word.lemma
            if len(w) > 0:
                current_sentence.append(w.lower())
                current_tok_sentence.append(w.lower())
                current_tok_lemma_sentence.append(l.lower())
                if not word.text in stop_words:
                    current_tok_stop_sentence.append(w.lower())
                    current_tok_lemma_stop_sentence.append(l.lower())
        
        sent = ' '.join(current_sentence)
        raw_tok_sentences.append(current_raw_tok_sentence)
        all_sentences.append(sent.lower())
        all_tok_sentences.append(current_tok_sentence)
        all_tok_stop_sentences.append(current_tok_stop_sentence)
        all_tok_lemma_sentences.append(current_tok_lemma_sentence)
        all_tok_lemma_stop_sentences.append(current_tok_lemma_stop_sentence)
    return all_sentences, all_tok_sentences, all_tok_stop_sentences, all_tok_lemma_sentences, all_tok_lemma_stop_sentences, raw_tok_sentences


def get_correct_answers(df, stop_words):
    correct_answers = []
    correct_answers_parsed = []
    correct_answers_parsed_tok = []
    correct_answers_parsed_stop = []
    correct_answers_parsed_lemma = []
    correct_answers_parsed_lemma_stop = []
    correct_answers_raw = []
    correct_answers_loc = []
    answer_reformulations = []
    questions_parsed = []
    questions_parsed_tok = []
    questions_parsed_stop = []
    questions_parsed_lemma = []
    questions_parsed_lemma_stop = []
    questions_raw = []
    # parse out the correct answer in the choices column
    for index, row in df.iterrows():
        answers = row['choices']
        df_row = pd.DataFrame(answers)
        # Collect the correct answer, add it to list. Save refomulation for those that use it
        answer_row = df_row.loc[df_row['type'] == 'Correct answer']
        answer_reformulation = False
        if answer_row.iloc[0]['extra']:
            answer_reformulation = answer_row.iloc[0]['extra']['comment']
        
        # parse the answer the same way as the context is parsed.
        correct_answer = answer_row.iloc[0]['text']
        doc = nlp(correct_answer)
        d_1, d_2, d_3, d_4, d_5, d_6 = make_sentences_from_tokens(doc, stop_words)
        correct_answer_parsed = d_1[0]
        correct_answer_parsed_tok = d_2[0]
        correct_answer_parsed_stop = d_3[0]
        correct_answer_parsed_lemma = d_4[0]
        correct_answer_parsed_lemma_stop = d_5[0]
        correct_answer_raw = d_6[0]

        # parse the question the same way as the context is parsed.
        question_raw = row['question']
        q_doc = nlp(question_raw)
        q_1, q_2, q_3, q_4, q_5, q_6 = make_sentences_from_tokens(q_doc, stop_words)
        question_parsed = q_1[0]
        question_parsed_tok = q_2[0]
        question_parsed_stop = q_3[0]
        question_parsed_lemma = q_4[0]
        question_parsed_lemma_stop = q_5[0]
        question_raw = q_6[0]
        
        correct_answer_loc = answer_row.iloc[0]['start']
        correct_answers.append(correct_answer)
        correct_answers_parsed.append(correct_answer_parsed)
        correct_answers_parsed_tok.append(correct_answer_parsed_tok)
        correct_answers_parsed_stop.append(correct_answer_parsed_stop)
        correct_answers_parsed_lemma.append(correct_answer_parsed_lemma)
        correct_answers_parsed_lemma_stop.append(correct_answer_parsed_lemma_stop)
        correct_answers_loc.append(correct_answer_loc)
        correct_answers_raw.append(correct_answer_raw)
        
        answer_reformulations.append(answer_reformulation)
        
        questions_parsed.append(question_parsed)
        questions_parsed_tok.append(question_parsed_tok)
        questions_parsed_stop.append(question_parsed_stop)
        questions_parsed_lemma.append(question_parsed_lemma)
        questions_parsed_lemma_stop.append(question_parsed_lemma_stop)
        questions_raw.append(question_raw)
   
    return correct_answers, correct_answers_parsed, correct_answers_parsed_tok, correct_answers_parsed_stop, correct_answers_parsed_lemma, correct_answers_parsed_lemma_stop, correct_answers_loc, correct_answers_raw, answer_reformulations, questions_parsed, questions_parsed_tok, questions_parsed_stop, questions_parsed_lemma, questions_parsed_lemma_stop, questions_raw


def filter_out_reformulated(df):
    # Filter out all the rows where the answer is reformulated!!
    # This is, all the rows where 'answer_reformulation' in not False
    print('number of original questions: ', len(df))
    df = df[df['answer_reformulation'] == False]
    print('Number of remaining, after removing those with reformulation: ', len(df))
    return df

# check in which sentence the answer can be found

def collect_sentence_number_statistics(df, stop_words):
    idx_of_ans = []
    sentences_with_ans = []
    idx_of_ans_text = []
    total_num_sents = []
    ans_loc_frac = []
    all_context_sentences = []
    raw_context_tok_word_sentences = []
    all_context_tok_word_sentences = []
    all_context_tok_word_stop_sentences = []
    all_context_tok_lemma_sentences = []
    all_context_tok_lemma_stop_sentences = []
    for index, row in df.iterrows():
        # iterate over all characters in the paragraph and find in which sentence the location is
        tot_chars = 0
        answer = row['correct_answer_parsed']
        answer_loc = int(row['correct_answer_loc'])
        text = row['context']
        # split the text into each sentence
        doc = nlp(text)
        sentences, tok_word_sent, tok_word_sent_stop, tok_lemma_sent, tok_lemma_sent_stop, tok_word_raw = make_sentences_from_tokens(doc, stop_words)
        all_context_sentences.append(sentences)
        all_context_tok_word_sentences.append(tok_word_sent)
        all_context_tok_word_stop_sentences.append(tok_word_sent_stop)
        all_context_tok_lemma_sentences.append(tok_lemma_sent)
        all_context_tok_lemma_stop_sentences.append(tok_lemma_sent_stop)
        raw_context_tok_word_sentences.append(tok_word_raw)

        # find in which sentences the answer is. How to know if it is the answer to the correct question??
        found_indexes = []
        loc_idx = None
        sentence_with_ans = None
        for index, sent in enumerate(sentences):
            num_chars = len(sent)+1 # TODO: check how to do this correctly with the current parsing!!
            tot_chars += num_chars
            if not loc_idx and tot_chars > answer_loc: # only collect if not already found
                loc_idx = index
                sentence_with_ans = sent
            if answer in sent:
                found_indexes.append(index)
        if not loc_idx:
            # if did not find sentence with answer in the text, sentence must be at the end (with sentence parsing characters are removed)
            loc_idx = index
            sentence_with_ans = sent

        
        # Match the indexes with the indexes found in text
        if not loc_idx in found_indexes:
            if len(found_indexes) == 1:
                # replace with where the index was found in the text
                loc_idx = found_indexes[0]
                sentence_with_ans = sentences[loc_idx]
            elif len(found_indexes) > 1:
                diff = np.abs(np.array(found_indexes) - loc_idx)
                min_diff = np.min(diff)
                min_diff_idx = diff.tolist().index(min_diff)
                # replace the index with the one found in text that is closest
                loc_idx = found_indexes[min_diff_idx]
                sentence_with_ans = sentences[loc_idx]
            else:
                print('ALERT - answer not found!')
                print('sentence by index: ', sentence_with_ans)
                print('answer: ', answer)



        # append the found indexes to the array for all paragraphs
        idx_of_ans_text.append(found_indexes)
        sentences_with_ans.append(sentence_with_ans) # append the sentence with the correct answer
        idx_of_ans.append(loc_idx) # append the location of the answer!
        total_num_sents.append(len(sentences))
        fracs = loc_idx/len(sentences)
        ans_loc_frac.append(fracs)

    return idx_of_ans, sentences_with_ans, idx_of_ans_text, total_num_sents, ans_loc_frac, all_context_sentences, all_context_tok_lemma_sentences, all_context_tok_lemma_stop_sentences, all_context_tok_word_stop_sentences, all_context_tok_word_sentences, raw_context_tok_word_sentences


def main(args):
    df = pd.read_pickle(args.data_path)

    stop_words = define_stopwords()

    df['correct_answer'], df['correct_answer_parsed'], df['correct_answer_parsed_tok'], df['correct_answer_parsed_stop'], \
    df['correct_answer_parsed_lemma'], df['correct_answer_parsed_lemma_stop'], df['correct_answer_loc'], \
    df['correct_answer_raw'], df['answer_reformulation'], df['question_parsed'],  df['question_parsed_tok'], \
    df['question_parsed_stop'], df['question_parsed_lemma'], df['question_parsed_lemma_stop'], df['question_raw'] = get_correct_answers(df, stop_words)

    df = filter_out_reformulated(df)

    df['answer_location'], df['sent_with_ans'], df['answer_locations_text'], df['paragraph_len'], df['loc_frac'], \
    df['context_parsed'], df['context_parsed_tok_lemma'], df['context_parsed_tok_lemma_stop'], \
    df['context_parsed_tok_stop'], df['context_parsed_tok'], df['context_raw']  = collect_sentence_number_statistics(df, stop_words)


    # save dataframe
    df = df.reset_index() # reset index so that its indexed from 0 to max
    df.to_pickle(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('output_path', type=str,
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--single', dest='single_file', action='store_true')


    args = parser.parse_args()
    main(args)

