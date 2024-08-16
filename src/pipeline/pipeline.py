from strategy import EmbeddingStrategy
from typing import List

class Pipeline():
    def __init__(self) -> None:
        self.handlers: List[EmbeddingStrategy] = []