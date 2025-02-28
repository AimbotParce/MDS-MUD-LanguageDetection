from typing import Iterable, List

from . import Tokenizer


class BigramTokenizer(Tokenizer):
    """
    Given a sentence, it returns a list of all the pairs of letters that appear one after the other.
    Keep in mind that the count of bigrams is always one less than the count of characters in a sentence.
    This tokenizer does not remove whitespaces, as they may be important (some languages might have different
    probabilities for words starting or ending in some letters).
    """
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return [list(tuple(map("".join, zip(sentence, sentence[1:])))) for sentence in data]

    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

