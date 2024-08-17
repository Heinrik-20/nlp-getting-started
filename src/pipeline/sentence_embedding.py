import pandas as pd

from .strategy import EmbeddingStrategy
from .cleaning import Cleaning
from sentence_transformers import SentenceTransformer

class SentenceEmbedding(EmbeddingStrategy):
    def __init__(self, cleaning_cls: Cleaning) -> None:
        
        self.cleaning_cls = cleaning_cls
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        super(SentenceEmbedding, self).__init__(cleaning_cls)
        return
    
    def generate_embedding(self):

        print("Preparing data")
        self.cleaning_cls()

        print("Constructing sentences from documents")
        train_corpus = [" ".join(doc) for doc in self.cleaning_cls.train_corpus]
        test_corpus = [" ".join(doc) for doc in self.cleaning_cls.test_corpus]

        print("Encoding sentences")
        train_embeddings = self.model.encode(train_corpus)
        test_embeddings = self.model.encode(test_corpus)

        print("Preparing dataframe")
        columns = [f"mpnet_embedding_{i}" for i in range(len(train_embeddings[0]))]
        train_df = pd.DataFrame(data=train_embeddings, columns=columns)
        test_df = pd.DataFrame(data=test_embeddings, columns=columns)
        train_df['target'] = self.cleaning_cls.train_df['target']

        return train_df, test_df