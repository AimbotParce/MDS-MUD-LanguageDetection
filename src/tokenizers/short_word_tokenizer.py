from typing import Iterable, List

from . import Tokenizer


class ShortWordTokenizer(Tokenizer):
    """
    Separates the sentences into lists of words, using whitespaces as separators.
    This differs from WordTokenizer in that if the word is too long, it will split
    it into singe characters.
    """
    max_word_length = 10

    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        res:List[List[str]] = []
        for sent in data: # Word tokenizer from nltk is language-dependent. Cannot use it
            sent_res:List[str] = []
            for word in str.split(sent):
                if len(word) > self.max_word_length:
                    sent_res.extend(list(word))
                else:
                    sent_res.append(word)
            res.append(sent_res)
        return res


    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

