import nltk
import spacy
import re

from nltk.tokenize import word_tokenize, sent_tokenize



#TODO: Experiment with different tokenizers
    

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

def remove_stopwords():
    pass

# Rule based suffix removal
def perform_lemmatisation():  # british spelling
    pass

# Dictionary linguistical database 
def perform_stemming():
    pass

def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Step 1: Perform URL Regex removal
    # Not needed probably --  Dani: I think we should remove URLs
    _sentence = sentence.apply(remove_urls)
    
    # Step 2: Perform tokenization
    _sentence = _sentence.apply(word_tokenize)

    # Step 3: Remove capitalization
    _sentence = _sentence.str.lower()


    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.
    return _sentence,labels



