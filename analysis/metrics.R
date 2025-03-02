library(readr)
library(FactoMineR)
library(mice)

raw <- read.csv("data/reports.csv")
summary(raw)
raw$dataset <- as.factor(raw$dataset)
raw$tokenizer <- as.factor(raw$tokenizer)
raw$vectorizer <- as.factor(raw$vectorizer)
raw$classifier <- as.factor(raw$classifier)
raw$remove_diacritics <- as.factor(raw$remove_diacritics)
raw$remove_urls <- as.factor(raw$remove_urls)
raw$remove_symbols <- as.factor(raw$remove_symbols)
raw$split_sentences <- as.factor(raw$split_sentences)
raw$lower <- as.factor(raw$lower)
raw$lemmatize <- as.factor(raw$lemmatize)
raw$stemmatize <- as.factor(raw$stemmatize)

# Single-value analysis
hist(raw$f1_micro)
hist(raw$f1_macro)
hist(raw$f1_weighted)
hist(raw$duration)

# Condes for f1_weighted
df <- raw[, -which(names(raw) %in% c("dataset", "vectorizer", "lemmatize", "stemmatize"))]
summary(df)
condes(df, num.var = which(names(df) == "f1_weighted"))

plot(df$test_coverage ~ df$tokenizer)
plot(df$f1_weighted ~ df$test_coverage)
plot(df$f1_weighted ~ df$tokenizer)
plot(df$f1_weighted ~ df$classifier)
boxplot(df$f1_weighted ~ df$tokenizer + df$classifier, col = df$classifier)
