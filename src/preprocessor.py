import re
from argparse import ArgumentParser
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download just in case
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

class Preprocessor(object):
    SYMBOLS_PATTERN = r"[\d,:;\"'(){}\[\]<>$€¥@#%^&*+=|]"
    URLS_PATTERN = r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?|www\.[a-zA-Z0-9.-]+(?:/[^\s]*)?"

    def __init__(
        self,
        remove_urls: bool = False,
        remove_symbols: bool = False,
        split_sentences: bool = False,
        lower: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        stemmatize: bool = False,
        stopword_lang: str = "english",
    ):
        self._remove_urls = remove_urls
        self._remove_symbols = remove_symbols
        self._split_sentences = split_sentences
        self._lower = lower

        self._remove_stopwords = remove_stopwords
        self._stopword_lang = stopword_lang
        self._lemmatize = lemmatize
        self._stemmatize = stemmatize


        self.stop_words = set()
        if self._remove_stopwords:
            self.stop_words = set(stopwords.words(self._stopword_lang))

    def _requires_tokenization(self) -> bool:
        # This tokenization is different from the one in the tokenizer.
        # This tokenization will always be word-wise, and is used to facilitate some of the preprocessing steps.
        return self._remove_stopwords or self.lemmatize or self._stemmatize

    def apply(self, sentences: Iterable[str], labels: Iterable[str]) -> tuple[list[str], list[str]]:
        """
        Given a list of sentences, apply all the required preprocessing steps
        to clean them, and transform them into something our vectorizer + classifier
        can work with.

        This object performs steps such as sentence splitting, URL removal,
        stopword removal, lemmatization, stemming, and more.

        Args
        ----
        sentences: Iterable[str]
            Input sentences to preprocess
        labels: Iterable[str]
            Input labels to preprocess

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing the preprocessed sentences and labels
        """
        x = sentences
        y = labels

        if self._remove_urls:  # Step 1: Perform URL Regex matching removal
            x = map(self.remove_urls, x)

        if self._remove_symbols:  # Step 2: Perform Number Regex matching removal
            x = map(self.remove_numbers_and_symbols, x)

        if self._split_sentences:  # Step 3: Perform sentence splitting (punkt)
            _x = map(sent_tokenize, x) # Uses Punkt by default # TODO: Sentence tokenize is language-dependent!!!!!
            x, y = self._flatten_sentences(_x, y)

        if self._lower:  # Step 5: Remove capitalization
            x = map(lambda s: s.lower(), x)

        if self._requires_tokenization():  # Step 4: Tokenization
            tokens = map(word_tokenize, x)  # Tokenize TODO: Word tokenize is language-dependent!!!!!

            if self._remove_stopwords:  # Step 6: Stopword removal (optional and not recommendable)
                # TODO BROKEN: somehow needs to parse language into stopword function
                tokens = map(self.remove_stop_words, tokens)

            if self._lemmatize:  # Step 7: Lematization (optional)
                tokens = map(self.lemmatize, tokens)

            if self._stemmatize:  # Step 8: Stemming (optional)
                tokens = map(self.stem, tokens)

            x = map(" ".join, tokens)

        return list(x), list(y)
    
    
    @staticmethod
    def _flatten_sentences(sentences: list[list[str]], labels: list[str]) -> tuple[list[str], list[str]]:
        res_sentences = []
        res_labels = []
        for sentences, label in zip(sentences, labels):
            for sentence in sentences:
                res_sentences.append(sentence)
                res_labels.append(label)
        return res_sentences, res_labels


    @staticmethod
    def remove_urls(text: str) -> str:
        # Matches HTTP(S) and WWW URLs
        return re.sub(Preprocessor.URLS_PATTERN, "", text)

    @staticmethod
    def remove_numbers_and_symbols(text: str) -> str:
        return re.sub(Preprocessor.SYMBOLS_PATTERN, "", text)


    def remove_stop_words(self, tokens: Iterable[str]) -> list[str]:
        return list(filter(lambda token: token not in self.stop_words, tokens))

    @staticmethod
    def lemmatize(tokens: Iterable[str]) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        return list(map(lemmatizer.lemmatize, tokens))

    @staticmethod
    def stem(tokens: Iterable[str]) -> list[str]:
        stemmer = PorterStemmer()
        return list(map(stemmer.stem, tokens))
    
    
if __name__ == "__main__":
    # Benchmark
    import timeit
    from pathlib import Path

    import pandas as pd

    raw = pd.read_csv(Path(__file__).parent.parent / "data" / "dataset.csv")
    preprocessor = Preprocessor(remove_urls=True, remove_symbols=True, split_sentences=True,
                       lower=True, remove_stopwords=True, lemmatize=True, stemmatize=True)

    ATTEMPTS = 8
    time = timeit.timeit(lambda: preprocessor.apply(raw['Text'], raw['language']), number=ATTEMPTS)
    print("Done in", time/ATTEMPTS, "seconds per execution on average.")