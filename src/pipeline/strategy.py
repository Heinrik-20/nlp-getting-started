from __future__ import annotations

import pandas as pd
import re
import string
import nltk

from abc import ABC, abstractmethod
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List

class EmbeddingStrategy(ABC):
    def __init__(self, train_path: str, test_path: str) -> None:
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        self.train_corpus = None
        self.test_corpus = None
        self.dictionary = None

        self.train_bow = None
        self.test_bow = None
        self.all_bow = None 
        self.id2word = None

        super().__init__()

        return

    def cleaning_data(self):

        url_regex = r"https?:\/\/[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,4}\/([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        non_alphanumeric = r"[^a-zA-Z ]"
        stopword_and_punctuations = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation) | {''}

        def create_corpus(df) -> list:

            corpus = df['text'].apply(lambda x: re.sub(url_regex, "", x)).apply(lambda x: re.sub(non_alphanumeric, "", x)).str.lower().to_list()
            cleaned = []
            lemmatizer = WordNetLemmatizer()
            for doc in corpus:
                cleaned.append([lemmatizer.lemmatize(token.lower()) for token in doc.split(" ") if token.lower() not in stopword_and_punctuations])

            return cleaned

        cleaned_train = create_corpus(self.train_df)
        cleaned_test = create_corpus(self.test_df)

        self.dictionary = Dictionary(cleaned_train + cleaned_test)
        self.train_corpus = cleaned_train
        self.test_corpus = cleaned_test

        self.train_bow = [self.dictionary.doc2bow(doc) for doc in self.train_corpus]
        self.test_bow = [self.dictionary.doc2bow(doc) for doc in self.test_corpus]

        self.all_bow = self.train_bow + self.test_bow

        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token

        return

    @abstractmethod
    def generate_embedding(self,):
        raise NotImplementedError
