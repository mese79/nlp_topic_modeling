import pickle
from pprint import pprint

import pandas as pd
import nltk
from gensim.models import LdaModel
from gensim.corpora import Dictionary


def main():
    # load docs
    with open('data/documents.pkl', 'rb') as f:
        docs = pickle.load(f)
    # load corpus and dictionary
    with open('data/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    dictionary = Dictionary.load('data/dictionary.pkl')

    # load model
    lda_model: LdaModel = LdaModel.load('saved_models/lda_model.dat')

    # show topics' top words
    for t in range(lda_model.num_topics):
        words = ', '.join([w for w, p in lda_model.show_topic(t, topn=10)])
        print(f'\ntopic id: {t}\n\t{words}')


    # get topic for some doc
    doc_idx = 0
    df_papers = pd.read_csv('data/papers.csv')
    paper_ids = pd.read_csv('data/sel_paper_ids.csv').to_numpy()[0]
    sel_paper = df_papers[df_papers['id'] == paper_ids[doc_idx]]
    print(f'\n\npaper title: {sel_paper["title"].item()}')
    print(docs[doc_idx][:50])
    topics = lda_model.get_document_topics(corpus[doc_idx])
    print(f'\ntopics: {topics}')






if __name__ == '__main__':
    main()
