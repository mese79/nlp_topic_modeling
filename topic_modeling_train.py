import string
import logging
import pickle
from pprint import pprint
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')


def main():
    # There is over 6500 papers, takes too long to run LDA on all papers.
    df_papers = pd.read_csv('data/papers.csv').sample(2000, random_state=7)
    # save selection id for later testing
    df_papers['id'].to_csv('data/sel_paper_ids.csv', index=False)
    # to lower case and remove punctuations
    print('to lower case and remove punctuations...')
    df_papers = df_papers['paper_text'].str.lower()
    df_papers = df_papers.str.replace(f'[{string.punctuation}]', '', regex=True).tolist()

    # tokenize each document(row) into words
    print('tokenize documents...')
    tokenizer = RegexpTokenizer(r'\w+')
    docs = [tokenizer.tokenize(p) for p in df_papers]

    # remove stop words and numbers(keep words containing numbers)
    print('remove stop words, numbers and one char. tokens and lemmatize tokens...')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'et', 'al'])
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    docs = [
        [lemmatizer.lemmatize(token) for token in doc
            if token not in stop_words and not token.isnumeric() and len(token) > 1]
        for doc in docs
    ]

    # add bigrams into documents
    print('add bigrams into documents...')
    bigram = Phrases(docs, min_count=20)  # build bigram model
    for idx in range(len(docs)):
        docs[idx].extend([
            bi_token for bi_token in bigram[docs[idx]]
            if '_' in bi_token and bi_token not in docs[idx]
        ])

    # save documents
    with open('data/documents.pkl', 'wb') as f:
        pickle.dump(docs, f)

    # create a dictionary for documents
    print('create dictionary and Bag-Of-Words corpus...')
    dictionary = Dictionary(docs)
    # filter out words that occur in less than 20 documents,
    # or more than 60% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.9)
    # create Bag-Of-Words representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print(f'\nNumber of unique tokens: {len(dictionary)}')
    print(f'Number of documents: {len(corpus)}\n')
    # save dictionary and corpus
    dictionary.save('data/dictionary.pkl')
    with open('data/corpus.pkl', mode='wb') as f:
        pickle.dump(corpus, f)

    # train an LDA model
    print('\ntraining an LDA model...')
    # set training parameters:
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 300
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    lda_model = LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        workers=3,
        chunksize=chunksize,
        alpha='symmetric',  # auto
        eta='auto',
        passes=passes,
        iterations=iterations,
        eval_every=eval_every
    )
    # save model
    lda_model.save('saved_models/lda_model.dat')
    # Average topic coherence is the sum of topic coherences of all topics,
    # divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    # print('\nAverage topic coherence: %.4f.\n' % avg_topic_coherence)

    print('\ncalculating coherence score...')
    coherence_type = 'c_v'
    coherence_model = CoherenceModel(
        model=lda_model, texts=docs, dictionary=dictionary, coherence=coherence_type
    )
    coherence_score = coherence_model.get_coherence()
    print(f'Coherence Score ({coherence_type}): {coherence_score:.3f}\n')

    # visualize LDA results
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_vis.html')






if __name__ == '__main__':
    main()
