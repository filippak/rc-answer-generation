import pandas as pd
import stanza
import re
import argparse

stanza.download('sv', processors='tokenize,pos,lemma,depparse')
nlp = stanza.Pipeline(lang='sv', processors='tokenize,pos,lemma,depparse')

def add_words_to_corpus(doc, context_corpus):
    for sentence in doc.sentences:
        for raw_word in sentence.words:
            # only add if character is letter or number (removes , . ? ! etc.)
            w_r  = re.sub('[^\s]', '', raw_word.text) # only remove space, escape char etc.
            if not w_r.isnumeric():
                context_corpus.add(w_r.lower())
            w_1  = re.sub('[^\sa-zåäöA-ZÅÄÖ0-9_-]', '', raw_word.text) # braod definition of words, including numbers, _ and -
            w_2  = re.sub('[^\sa-zåäöA-ZÅÄÖ]', '', raw_word.text)
            if len(w_1) > 0 and not w_1.isnumeric():
                context_corpus.add(w_1.lower())
            if len(w_2) > 0:
                context_corpus.add(w_2.lower())
            word_lemma = str(raw_word.lemma)
            if word_lemma != raw_word.text and not word_lemma.isnumeric():
                context_corpus.add(word_lemma.lower())
    return context_corpus

def add_context_words(df):
    print('adding context words to corpus..')
    context_corpus = set()
    for index, row in df.iterrows():
        context = row['context']
        context_parsed = nlp(context)
        add_words_to_corpus(context_parsed, context_corpus)
    return context_corpus

# save context corpus to file
def save_context_corpus(filename, list):
    list.sort()
    with open(filename, 'w') as out:
        for word in list:
            out.write(word + '\n')

def main(args):
    df = df = pd.read_pickle(args.data_path)
    context_corpus = add_context_words(df)
    context_corpus_list = list(context_corpus)
    
    save_context_corpus(args.output_path, context_corpus_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')

    args = parser.parse_args()
    main(args)
