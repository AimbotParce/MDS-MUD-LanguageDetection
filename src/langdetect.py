import argparse
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import classifiers
import preprocessor
import tokenizers
import vectorizers
from utils import (compute_coverage, computePCA, normalizeData,
                   plot_Confusion_Matrix, plotPCA)

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide-plots", help="Hide plots", action="store_true")
    parser.add_argument("--report-results", help="Path to report the results to", type=Path, default=None)
    parser.add_argument(
        "-i",
        "--input",
        help="Input data in csv format",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "dataset.csv",
    )
    parser.add_argument("-v", "--voc_size", help="Vocabulary size", type=int, default=2000)
    parser.add_argument(
        "-t", "--tokenizer", help="Kind of tokenizer to use", choices=tokenizers.options, default="char"
    )
    parser.add_argument(
        "-c", "--classifier", help="Kind of classifier to use", choices=classifiers.options, default="nb"
    )
    parser.add_argument(
        "--vectorizer", help="Kind of vectorizer to use", choices=vectorizers.options, default="token-count"
    )

    # Preprocessing options
    parser.add_argument("--remove-urls", help="Remove URLs", action="store_true")
    parser.add_argument("--remove-symbols", help="Remove Symbols", action="store_true")
    parser.add_argument("--split-sentences", help="Split Sentences", action="store_true")
    parser.add_argument("--lower", help="Lowercase", action="store_true")
    parser.add_argument("--lemmatize", help="Lemmatize", action="store_true")
    parser.add_argument("--stemmatize", help="Stemmatize", action="store_true")
    return parser


if __name__ == "__main__":
    start = time.time()
    parser = get_parser()
    args = parser.parse_args()

    INPUT: Path = args.input
    VOC_SIZE: int = args.voc_size
    TOKENIZER: str = args.tokenizer
    VECTORIZER: str = args.vectorizer
    CLASSIFIER: str = args.classifier
    HIDE_PLOTS: bool = args.hide_plots
    REPORT_RESULTS: Optional[Path] = args.report_results

    # Preprocessing options
    REMOVE_URLS: bool = args.remove_urls
    REMOVE_SYMBOLS: bool = args.remove_symbols
    SPLIT_SENTENCES: bool = args.split_sentences
    LOWER: bool = args.lower
    LEMMATIZE: bool = args.lemmatize
    STEMMATIZE: bool = args.stemmatize

    print("========")
    print("Parameters:")
    print("Input:", INPUT)
    print("Vocabulary size:", VOC_SIZE)
    print("Tokenizer:", TOKENIZER)
    print("Vectorizer:", VECTORIZER)
    print("Classifier:", CLASSIFIER)

    print("Preprocessing options:")
    print("Remove URLs:", REMOVE_URLS)
    print("Remove Symbols:", REMOVE_SYMBOLS)
    print("Split Sentences:", SPLIT_SENTENCES)
    print("Lowercase:", LOWER)
    print("Lemmatize:", LEMMATIZE)
    print("Stemmatize:", STEMMATIZE)
    print("========")

    print("Reading data...", end=" ")
    raw = pd.read_csv(INPUT)
    print("Done!")

    # Languages
    languages = set(raw["language"])
    print("Languages", languages)
    print("========")

    # Split Train and Test sets
    print("Splitting data...", end=" ")
    X_train, X_test, y_train, y_test = train_test_split(raw["Text"], raw["language"], test_size=0.2, random_state=seed)
    print("Done!")

    print("Split sizes:")
    print("Train:", len(X_train))
    print("Test:", len(X_test))
    print("========")

    # Preprocess text (Word granularity only)
    start = time.time()
    print("Preprocessing text...", end=" ", flush=True)
    preprocessor = preprocessor.Preprocessor(
        remove_urls=REMOVE_URLS,
        remove_symbols=REMOVE_SYMBOLS,
        split_sentences=SPLIT_SENTENCES,
        lower=LOWER,
        lemmatize=LEMMATIZE,
        stemmatize=STEMMATIZE,
    )
    X_train_pre, y_train = preprocessor.apply(X_train, y_train)
    X_test_pre, y_test = preprocessor.apply(X_test, y_test)
    print(f"Done! ({time.time()-start:.1f}s)", flush=True)

    # Tokenize text
    start = time.time()
    print("Tokenizing text...", end=" ", flush=True)
    tokenizer = tokenizers.getTokenizer(TOKENIZER)
    X_train_tok = tokenizer.fit_transform(X_train_pre)
    X_test_tok = tokenizer.transform(X_test_pre)
    print(f"Done! ({time.time()-start:.1f}s)", flush=True)

    # Compute text features
    start = time.time()
    print("Computing text features...", end=" ", flush=True)
    vectorizer = vectorizers.getVectorizer(VECTORIZER, max_features=VOC_SIZE)
    X_train_vec = vectorizer.fit_transform(X_train_tok)
    X_test_vec = vectorizer.transform(X_test_tok)
    vocab = vectorizer.get_vocab()
    print(f"Done! ({time.time()-start:.1f}s)", flush=True)

    print("Number of tokens in the vocabulary:", len(vocab))
    print("Vocabulary:", ", ".join(vocab[:10]), "...")
    test_coverage = compute_coverage(vocab, X_test_tok)
    train_coverage = compute_coverage(vocab, X_train_tok)
    print(f"Coverage on train: {train_coverage*100:.2f}% and test: {test_coverage*100:.2f}%")
    print("========")

    # Normalize Data
    X_train, X_test = normalizeData(X_train_vec, X_test_vec)

    # Apply classification algorithms
    start = time.time()
    print("Fitting classifier...", end=" ", flush=True)
    classifier = classifiers.getClassifier(CLASSIFIER)
    classifier.fit(X_train, y_train)
    print(f"Done! ({time.time()-start:.1f}s)", flush=True)

    # Predict
    start = time.time()
    print("Predicting...", end=" ", flush=True)
    y_predict = classifier.predict(X_test)
    print(f"Done! ({time.time()-start:.1f}s)", flush=True)

    print("Prediction Results:")
    f1_micro = f1_score(y_test, y_predict, average="micro")
    f1_macro = f1_score(y_test, y_predict, average="macro")
    f1_weighted = f1_score(y_test, y_predict, average="weighted")
    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))

    pca = computePCA(X_train)
    print("Variance explained by PCA:", pca.explained_variance_ratio_)
    print("========")

    if REPORT_RESULTS is not None:
        print("Writing results to", REPORT_RESULTS)
        with open(REPORT_RESULTS, "a") as f:
            f.write(
                f'"{INPUT}",{VOC_SIZE},"{TOKENIZER}","{VECTORIZER}","{CLASSIFIER}",{REMOVE_URLS},{REMOVE_SYMBOLS},'
                f"{SPLIT_SENTENCES},{LOWER},{LEMMATIZE},{STEMMATIZE},{len(X_train_pre)},"
                f"{len(X_test_pre)},{len(vocab)},{train_coverage},{test_coverage},{float(f1_micro)},{float(f1_macro)},{float(f1_weighted)},"
                f"{float(pca.explained_variance_ratio_[0])},{time.time()-start}\n"
            )

    if HIDE_PLOTS:
        print("Plots are hidden")
    else:
        plot_Confusion_Matrix(y_test, y_predict, "Greens")

        print("PCA and Explained Variance:")
        plotPCA(pca, X_test, y_test, languages)
        print("========")
