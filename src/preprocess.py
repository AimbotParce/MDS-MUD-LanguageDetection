import nltk
import spacy
import re
import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


#Download just in case
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def remove_urls(text):
    # Matches HTTP(S) and WWW URLs
    url_pattern = r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?|www\.[a-zA-Z0-9.-]+(?:/[^\s]*)?"
    return re.sub(url_pattern, "", text)

def remove_numbers_and_symbols(text):
    sym_pattern = re.sub(r"[\d,:;\"'(){}\[\]<>$€¥@#%^&*+=|]", "", text)
    return re.sub(sym_pattern, "", text)

def sentence_splitting(text):
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

def preprocess(sentence, labels, **kwargs):
    """
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    """
    
    # Parse boolean action activation parameters for experimentation
    rm_urls = kwargs.get("remove_urls", True)
    rm_sim = kwargs.get("remove_symbols", True)
    split_sntcs = kwargs.get("split_sentences", True)
    tokenize = kwargs.get("tokenize", True)
    lower = kwargs.get("lower", True)
    remove_stopwords = kwargs.get('remove_stopwords', False)
    lemmatize = kwargs.get('lemmatize', False)
    stemmatize = kwargs.get('stemmatize', False)
    stopword_lang = kwargs.get('stopword_lang', "english")
    #tokenizer = kwargs.get("tokenizer")    

    # Step 1: Perform URL Regex matching removal
    if rm_urls:
        _sentence = sentence.apply(remove_urls)
    
    # Step 2: Perform Number Regex matching removal
    if rm_sim:
        _sentence = _sentence.apply(remove_numbers_and_symbols)

    # Step 3: Perform sentence splitting (punkt)
    if split_sntcs:
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

    # Step 4: Perform tokenization (word)
    #TODO: Experiment with different tokenizers
    if tokenize:
        _sentence = _sentence.apply(word_tokenize)

    # Step 5: Remove capitalization
    if lower:
        _sentence = _sentence.str.lower()

    # TODO BROKEN: somehow needs to parse language into stopword function
    # Step 6: Stopword removal (optional and not recommendable)
    if remove_stopwords:
        _sentence = _sentence.apply(lambda x: remove_stopwords(x, stopword_lang))
    

    # Step 7: Lematization (optional)
    if lemmatize:
        _sentence = _sentence.apply(perform_lemmatization)

    # Step 8: Stemming (optional)
    if stemmatize:
        _sentence = _sentence.apply(perform_stemming)

    return _sentence, _labels
