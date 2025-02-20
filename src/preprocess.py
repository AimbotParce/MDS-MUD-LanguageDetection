import nltk
import spacy
import re
import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer



#TODO: Experiment with different tokenizers

nltk.download("stopwords")
nltk.download("wordnet")


def remove_urls(text):
    # Matches HTTP(S) and WWW URLs
    url_pattern = r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?|www\.[a-zA-Z0-9.-]+(?:/[^\s]*)?"
    return re.sub(url_pattern, '', text)

def sentence_splitting(text):
    # Just in case of not having it
    nltk.download("punkt")

    if not isinstance(text, str):
        return text
    sentences = sent_tokenize(text)  # Uses Punkt by default
    return sentences

def remove_stopwords(tokens, language):
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token not in stop_words]

def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def perform_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
    pass

def preprocess(sentence, labels, **kwargs):
    """
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    """
    do_stopwords = kwargs.get('remove_stopwords', False)
    do_lemmatize = kwargs.get('do_lemmatize', False)
    do_stemming = kwargs.get('do_stemmin', False)
    stopword_lang = kwargs.get('stopword_lang', "english")

    # Step 1: Perform URL Regex removal
    _sentence = sentence.apply(remove_urls)

    # Step 2: Perform sentence splitting (punkt)
    _sentence = _sentence.apply(sentence_splitting)

    new_sentences = []
    new_labels = []

    for sent, label in zip(_sentence, labels):
        if isinstance(sent, list):
            new_sentences.extend(sent)
            new_labels.extend([label] * len(sent))
        else:  # There was no splitting
            new_sentences.append(sent)
            new_labels.append(label)

    # Convert from list back to pd Series
    _sentence = pd.Series(new_sentences)
    _labels = new_labels

    # Step 3: Perform tokenization (word)
    _sentence = _sentence.apply(word_tokenize)

    # Step 4: Remove capitalization
    _sentence = _sentence.str.lower()

    # Step 5: Lematisation (optional)
    # if do_stopwords:
    #     _sentence = _sentence.apply(remove_stopwords)
    # TODO BROKEN: somehow needs to parse language into stopword function
    # Still tho, I don't really see a need to remove stopwords for langdetect?

    # Step 6: Lematisation (optional)
    if do_lemmatize:
        _sentence = _sentence.apply(perform_lemmatization)

    # Step 7: Stemming (optional)
    if do_stemming:
        _sentence = _sentence.apply(perform_stemming)

    return _sentence, _labels
