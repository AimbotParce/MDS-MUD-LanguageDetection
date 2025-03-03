# MDS-MUD-LanguageDetection
Language Detection Practical

### Warning: `reports.csv` "duration" column is incorrect and meaningless.


Run the model with these commands:

## Best performance (paper results)

```bash
# Best overall (5000 vocab size)
python src/langdetect.py --tokenizer bigram --classifier mlp --voc-size 5000

# Best with 2000 vocabulary size
python src/langdetect.py --tokenizer bigram --classifier mlp --voc-size 2000 --remove-urls --lower --split-sentences
```

## Other options

```
# More tokenizers
--tokenizer word
--tokenizer char
--tokenizer short-word

# Other classifiers
--classifier dt
--classifier knn
--classifier rf
--classifier nb
--classifier lr
--classifier lda
--classifier svm

# Preprocessing options
--remove-diacritics
--remove-urls
--remove-symbols
--lower
--lemmatize
--stemmatize
```

Default dataset: `data/dataset.csv` (override with `-i`)
