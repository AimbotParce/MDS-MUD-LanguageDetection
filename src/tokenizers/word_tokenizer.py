from typing import Iterable, List

from nltk.tokenize import word_tokenize

from . import Tokenizer


class WordTokenizer(Tokenizer):
    "Separates sentences into lists of tokens. Can be either a static tokenizer or a trainable one"
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return list(map(word_tokenize, data)) # TODO: Word tokenizer is language-dependent!!!!!!

    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

