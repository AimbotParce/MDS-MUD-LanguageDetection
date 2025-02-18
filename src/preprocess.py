import nltk
import spacy
import re

from nltk.tokenize import word_tokenize

# Just in case of not having it
nltk.download("punkt")


#TODO: Experiment with different tokenizers
    

def remove_urls(text):
    # Matches HTTP(S) and WWW URLs
    url_pattern = r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?|www\.[a-zA-Z0-9.-]+(?:/[^\s]*)?"
    return re.sub(url_pattern, '', text)

def sentence_splitting():
    pass

def remove_stopwords():
    pass

# Rule based suffix removal
def perform_lemmatization():
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
    # Not needed probably
    sentence = sentence.apply(remove_urls)  
    
    # Step 2: Perform tokenization
    sentence = sentence.apply(word_tokenize)

    # Step 3: Remove capitalization
    sentence = sentence.str.lower()


    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.
    return sentence,labels



