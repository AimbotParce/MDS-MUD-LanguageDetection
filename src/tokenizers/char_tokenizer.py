from typing import Iterable, List

from . import Tokenizer


class CharTokenizer(Tokenizer):
    "Separates the sentences into lists of characters, ignoring whitespaces"
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return list(map(lambda x: x.replace(" ", ""), data))

    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

