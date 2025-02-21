from typing import Iterable, List

from . import Tokenizer


class BigramTokenizer(Tokenizer):
    "Separates sentences into lists of tokens. Can be either a static tokenizer or a trainable one"
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return [list(zip(sentence, sentence[1:])) for sentence in data]

    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

