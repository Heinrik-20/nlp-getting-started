from __future__ import annotations

import pandas as pd
import re
import string
import nltk

from abc import ABC, abstractmethod
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List
from .cleaning import Cleaning

class EmbeddingStrategy(ABC):
    def __init__(self, cleaning_cls: Cleaning) -> None:

        self.cleaning_cls = cleaning_cls

        super().__init__()

        return

    @abstractmethod
    def generate_embedding(self,):
        raise NotImplementedError
