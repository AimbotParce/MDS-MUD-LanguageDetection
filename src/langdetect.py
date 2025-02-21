import argparse
import random
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import classifiers
import tokenizers
import vectorizers
from preprocessor import Preprocessor
from utils import (compute_coverage, normalizeData, plot_Confusion_Matrix,
                   plot_F_Scores, plotPCA)

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=Path, 
                        default=Path(__file__).parent.parent / "data" / "dataset.csv")
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int,
                        default=2000)
    parser.add_argument("-t", "--tokenizer", help="Kind of tokenizer to use",
                        choices=tokenizers.options, default="char")
    parser.add_argument("-c", "--classifier", help="Kind of classifier to use",
                        choices=classifiers.options, default="nb")
    parser.add_argument("--vectorizer", help="Kind of vectorizer to use",
                        choices=vectorizers.options, default="token-count")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    INPUT:Path = args.input
    VOC_SIZE:int = args.voc_size
    TOKENIZER:str = args.tokenizer
    VECTORIZER:str = args.vectorizer
    CLASSIFIER:str = args.classifier

    print('========')
    print('Parameters:')
    print('Input:', INPUT)
    print('Vocabulary size:', VOC_SIZE)
    print('Tokenizer:', TOKENIZER)
    print('Vectorizer:', VECTORIZER)
    print('Classifier:', CLASSIFIER)
    print('========')

    print('Reading data...', end=' ')
    raw = pd.read_csv(INPUT)
    print('Done!')
    
    # Languages
    languages = set(raw['language'])
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    print('Splitting data...', end=' ')
    X_train, X_test, y_train, y_test = train_test_split(raw['Text'], raw['language'], test_size=0.2, random_state=seed)
    print('Done!')
    
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    
    # Preprocess text (Word granularity only)
    start = time.time()
    print('Preprocessing text...', end=' ', flush=True)
    preprocessor = Preprocessor(remove_urls=True, remove_symbols=True, split_sentences=False, 
                       lower=True, remove_stopwords=False, lemmatize=False, stemmatize=False)
    X_train_pre, y_train = preprocessor.apply(X_train, y_train)
    X_test_pre, y_test = preprocessor.apply(X_test,y_test)
    print(f'Done! ({time.time()-start:.1f}s)', flush=True)

    
    #Tokenize text
    start = time.time()
    print('Tokenizing text...', end=' ', flush=True)    
    tokenizer = tokenizers.getTokenizer(TOKENIZER)
    X_train_tok = tokenizer.fit_transform(X_train_pre)
    X_test_tok = tokenizer.transform(X_test_pre)
    print(f'Done! ({time.time()-start:.1f}s)', flush=True)

    #Compute text features
    start = time.time()
    print('Computing text features...', end=' ', flush=True)
    vectorizer = vectorizers.getVectorizer(VECTORIZER, max_features=VOC_SIZE)
    X_train_vec = vectorizer.fit_transform(X_train_tok)
    X_test_vec = vectorizer.transform(X_train_tok)
    vocab = vectorizer.get_vocab()
    print(f'Done! ({time.time()-start:.1f}s)', flush=True)

    print('Number of tokens in the vocabulary:', len(vocab))
    print('Vocabulary:', ', '.join(vocab[:10]), '...')
    print('Coverage: ', compute_coverage(vocab, X_test_tok))
    print('========')

    #Normalize Data
    X_train, X_test = normalizeData(X_train_vec, X_test_vec)
    
    #Apply classification algorithms
    start = time.time()
    print('Fitting classifier...', end=' ', flush=True)
    classifier = classifiers.getClassifier(CLASSIFIER)
    classifier.fit(X_train, y_train)
    print(f'Done! ({time.time()-start:.1f}s)', flush=True)

    #Predict
    start = time.time()
    print('Predicting...', end=' ', flush=True)
    y_predict = classifier.predict(X_test)
    print(f'Done! ({time.time()-start:.1f}s)', flush=True)
    
    print('Prediction Results:')    
    plot_F_Scores(y_test, y_predict)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, "Greens") 


    #Plot PCA
    print('PCA and Explained Variance:') 
    plotPCA(X_train, X_test,y_test, languages) 
    print('========')
