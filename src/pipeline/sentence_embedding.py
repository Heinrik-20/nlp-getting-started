from .strategy import EmbeddingStrategy

class SentenceEmbedding(EmbeddingStrategy):
    def __init__(self, file_path: str) -> None:

        self.lda = None
        self.tfidf = None

        super(SentenceEmbedding, self).__init__(file_path)
        return
    
    def generate_embedding(self):
        return