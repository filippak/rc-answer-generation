
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import argparse

tfidf_transformer = TfidfTransformer()
cv = CountVectorizer()

def combine_docs(df, col):
    docs = []
    for index, row in df.iterrows():
        context = row[col]
        if isinstance(context[0], list):
            sents = []
            for sent in context:
                sent_str = ' '.join(sent)
                sents.append(sent_str)
            context = sents
        doc = ' '.join(context)
        docs.append(doc)
    return docs

def create_tf_df(docs):
    word_count_vector = cv.fit_transform(docs)
    tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names())
    return word_count_vector, tf

def create_idf_df(word_count_vector):
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names(), 'idf_weights':tfidf_transformer.idf_})
    return X, idf

def create_tf_idf_df(df, col):
    docs = combine_docs(df, col)
    word_count_vector, tf = create_tf_df(docs)
    X, idf = create_idf_df(word_count_vector)
    tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names())
    return tf_idf, tf, idf

def create_idf_sv_corpus(idf, idf_sv_df):
    idf_mod = idf.copy()
    for index, row in idf_mod.iterrows():
        word = row['feature_name']
        matching_row = idf_sv_df.loc[idf_sv_df['word'] == word]
        if not matching_row.empty:
            row['idf_weights'] = matching_row.iloc[0]['idf'] # replace the idf score by the one from the bigger corpus
    return idf_mod

def tf_idf_from_sv_idf(tf, idf):
    tf_idf = tf.copy()
    tf_idf = tf_idf.mul(idf['idf_weights'].values, axis=1)

    # normalize the tf_idf values
    sqrt_vec = np.sqrt(tf_idf.pow(2).sum(axis=1))
    tf_idf_norm = tf_idf.div(sqrt_vec, axis=0)
    return tf_idf_norm

def compute_on_bigger_corpus_idf(tf, idf):
    idf_sv_df = pd.read_csv("../../idf_sv.csv")
    idf_sv_corpus = create_idf_sv_corpus(idf, idf_sv_df)
    tf_idf_sv_corpus = tf_idf_from_sv_idf(tf, idf_sv_corpus)
    return tf_idf_sv_corpus


def main(args):
    df = pd.read_pickle(args.data_path)

    tf_idf, tf, idf = create_tf_idf_df(df, 'context_parsed')
    tf_idf_lemma_stop, tf_lemma_stop, idf_lemma_stop = create_tf_idf_df(df, 'context_parsed_tok_lemma_stop')

    # save dataframe
    path = args.output_path + '.pkl'
    tf_idf.to_pickle(path)

    path_lemma_stop = args.output_path + '_lemma_stop.pkl'
    tf_idf_lemma_stop.to_pickle(path_lemma_stop)

    tf_idf_sv_corpus = compute_on_bigger_corpus_idf(tf_lemma_stop, idf_lemma_stop)
    path_sv_corpus = args.output_path + '_lemma_stop_sv_corpus.pkl'
    tf_idf_lemma_stop.to_pickle(path_sv_corpus)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')


    args = parser.parse_args()
    main(args)