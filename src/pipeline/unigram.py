import pandas as pd
import numpy as np
import nltk

# Generating embeddings using an LDA model
from gensim.models import LdaModel
from gensim.models import TfidfModel
from .strategy import EmbeddingStrategy
from typing import List

# Typing
from typing import Optional, Tuple

nltk.download('stopwords')
nltk.download('wordnet')

class UnigramEmbedding(EmbeddingStrategy):
    def __init__(self, train_path: str, test_path: str) -> None:

        self.lda = None
        self.tfidf = None

        super(UnigramEmbedding, self).__init__(train_path, test_path)
        return
    
    def generate_embedding(self):
        
        print("Preparing data")
        self.cleaning_data()

        print("Generating LDA embeddiing")
        lda_train, lda_test = self.generate_lda()

        print("Generation TFIDF embedding")
        tfidf_train, tfidf_test = self.generate_tfidf()

        train_df = pd.concat([lda_train, tfidf_train], axis=1)
        train_df['target'] = self.train_df['target']

        test_df = pd.concat([lda_test, tfidf_test], axis=1)

        return train_df, test_df
    
    def generate_lda(
        self,
        num_topics: Optional[int]=10,
        chunksize: Optional[int]=3500,
        passes: Optional[int]=5,
        iterations: Optional[int]=400,
        eval_every: Optional[int]=None,
        seed: Optional[int]=100,
    ):

        lda = LdaModel(
            corpus=self.all_bow,
            id2word=self.id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
            random_state=seed,
        )

        columns = [f"Topic {i}" for i in range(num_topics)]
        train_values = []
        test_values = []

        for bow, values in zip([self.train_bow, self.test_bow], [train_values, test_values]):
            for topic_distr in lda.get_document_topics(bow):
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
        
        self.lda = lda

        return pd.DataFrame(data=train_values, columns=columns), pd.DataFrame(data=test_values, columns=columns)
    
    def generate_tfidf(self,):
        
        tfidf = TfidfModel(self.all_bow)
        unique_tokens = len(self.dictionary)

        train_values = []
        test_values = []
        columns = [f'Word_{i}' for i in range(unique_tokens)]
        
        for bow, values in zip([self.train_bow, self.test_bow], [train_values, test_values]):
            for text in bow:
                embedding = np.zeros(shape=(unique_tokens, ))

                if len(text):
                    output = np.array(tfidf[text])
                    embedding[output[:, 0].astype(np.int32)] = output[:, 1]
                    
                values.append(embedding)

        self.tfidf = tfidf

        return pd.DataFrame(data=train_values, columns=columns), pd.DataFrame(data=test_values, columns=columns)
    
    