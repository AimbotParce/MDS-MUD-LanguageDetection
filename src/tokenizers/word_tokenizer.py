from typing import Iterable, List

from . import Tokenizer


class WordTokenizer(Tokenizer):
    "Separates the sentences into lists of words, using whitespaces as separators"
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return list(map(str.split, data)) # Word tokenizer from nltk is language-dependent. Cannot use it here.

    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

