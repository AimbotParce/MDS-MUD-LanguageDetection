library(readr)
library(FactoMineR)
library(mice)
library(ggplot2)
library(dplyr)

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


dfs <- df %>%
    arrange(tokenizer, classifier) %>%
    mutate(Combined = factor(paste(tokenizer, classifier, sep = " - "), levels = unique(paste(tokenizer, classifier, sep = " - "))))

# Boxplot f1_weighted vs tokenizer and classifier
ggplot(dfs, aes(x = interaction(tokenizer, classifier, sep = " - "), y = f1_weighted, fill = classifier)) +
    geom_boxplot() +
    labs(x = "Tokenizer - Classifier", y = "F1 Weighted", fill = "Classifier") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_x_discrete(limits = unique(interaction(dfs$tokenizer, dfs$classifier, sep = " - ")))
ggsave("plots/f1_weighted_tokenizer_classifier.png")
