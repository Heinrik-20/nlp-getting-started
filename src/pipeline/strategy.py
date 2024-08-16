from __future__ import annotations

import pandas as pd
import re
import string
import nltk

from abc import ABC, abstractmethod
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer

class EmbeddingStrategy(ABC):
    def __init__(self, file_path: str) -> None:
        
        self.file_path = file_path

        self.corpus = None
        self.dictionary = None
        self.bow = None
        self.id2word = None

        super().__init__()

        return

    def cleaning_data(self):

        df = pd.read_csv(self.file_path)
    
        url_regex = r"https?:\/\/[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,4}\/([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        non_alphanumeric = r"[^a-zA-Z ]"
        stopword_and_punctuations = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation) | {''}

        corpus = df['text'].apply(lambda x: re.sub(url_regex, "", x)).apply(lambda x: re.sub(non_alphanumeric, "", x)).str.lower().to_list()
        cleaned = []
        lemmatizer = WordNetLemmatizer()
        for doc in corpus:
            cleaned.append([lemmatizer.lemmatize(token.lower()) for token in doc.split(" ") if token.lower() not in stopword_and_punctuations])

        del df

        self.dictionary = Dictionary(cleaned)
        self.corpus = cleaned

        self.dictionary.filter_extremes(no_below=10, no_above=0.5)
        self.bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]

        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token

        return

    @abstractmethod
    def generate_embedding(self,):
        raise NotImplementedError
