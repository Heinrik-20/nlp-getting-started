import pandas as pd
import numpy as np
import re
import string
import nltk

# Generating embeddings using an LDA model
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer

# Typing
from typing import Optional, Tuple

nltk.download('stopwords')
nltk.download('wordnet')


def generate_lda(
        num_topics: Optional[int]=10,
        chunksize: Optional[int]=3500,
        passes: Optional[int]=5,
        iterations: Optional[int]=400,
        eval_every: Optional[int]=None,
        seed: Optional[int]=100,
) -> Tuple[pd.DataFrame, LdaModel]:

    train_df = pd.read_csv("../data/train.csv")
    

    url_regex = r"https?:\/\/[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,4}\/([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    non_alphanumeric = r"[^a-zA-Z ]"
    symbols = string.punctuation
    stopword_and_punctuations = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation) | {''}

    corpus = train_df['text'].apply(lambda x: re.sub(url_regex, "", x)).apply(lambda x: re.sub(non_alphanumeric, "", x)).str.lower().to_list()
    cleaned = []
    target = train_df['target']
    lemmatizer = WordNetLemmatizer()
    for doc in corpus:
        cleaned.append([lemmatizer.lemmatize(token.lower()) for token in doc.split(" ") if token.lower() not in stopword_and_punctuations])

    del train_df

    word_dict = Dictionary(cleaned)
    word_dict.filter_extremes(no_below=10, no_above=0.5)
    corpus = [word_dict.doc2bow(doc) for doc in cleaned]

    _ = word_dict[0]
    id2word = word_dict.id2token

    lda = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    columns = [f"Topic {i}" for i in range(num_topics)]
    values = []

    for topic_distr in lda.get_document_topics(corpus):
        distri = []
        i, j = 0, 0
        topic_length = len(topic_distr)
        while i < num_topics:
            if topic_distr[j][0] != i:
                distri.append(0)
            else:
                distri.append(topic_distr[j][1])
                j = min(topic_length - 1, j + 1)
            i += 1
        values.append(distri)

    train_df = pd.DataFrame(data=values, columns=columns)
    train_df['target'] = target

    return train_df, lda