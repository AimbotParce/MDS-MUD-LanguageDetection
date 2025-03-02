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


shapiro.test(df$f1_weighted)

boxplot(f1_weighted ~ remove_diacritics,
    data = df,
    main = "F1 Weighted by Diacritics Removal",
    ylab = "F1 Weighted",
    col = c("skyblue", "lightgreen")
)
wilcox.test(f1_weighted ~ remove_diacritics, data = df)

boxplot(f1_weighted ~ remove_urls,
    data = df,
    main = "F1 Weighted by URL Removal",
    ylab = "F1 Weighted",
    col = c("salmon", "lightyellow")
)
wilcox.test(f1_weighted ~ remove_urls, data = df)

boxplot(f1_weighted ~ remove_symbols,
    data = df,
    main = "F1 Weighted by Symbols Removal",
    ylab = "F1 Weighted",
    col = c("lightblue", "lavender")
)
wilcox.test(f1_weighted ~ remove_symbols, data = df)


boxplot(f1_weighted ~ split_sentences,
    data = df,
    main = "F1 Weighted by Sentence Splitting",
    ylab = "F1 Weighted",
    col = c("lightgreen", "lightpink")
)
wilcox.test(f1_weighted ~ split_sentences, data = df)

boxplot(f1_weighted ~ lower,
    data = df,
    main = "F1 Weighted by Lowercasing",
    ylab = "F1 Weighted",
    col = c("lightcoral", "lightblue")
)
wilcox.test(f1_weighted ~ lower, data = df)

boxplot(f1_weighted ~ classifier,
    data = df,
    main = "F1 Weighted by Classifier",
    ylab = "F1 Weighted",
    col = rainbow(length(unique(df$classifier)))
)
kruskal.test(f1_weighted ~ classifier, data = df)


summary(df)
model <- glm(f1_weighted ~ remove_diacritics + remove_urls + remove_symbols + split_sentences + lower, data = df)
summary(model)

# Find the best combination of parameters, based on the highest F1 Weighted
best <- df[which.max(df$f1_weighted), ]
best
# Tokenizer: bigram
# Classifier: mlp
# Remove Diacritics: False
# Remove URLs: True
# Remove Symbols: False
# Split Sentences: True
# Lower: True
# ---
# Train Coverage: 0.8403668
# Test Coverage: 0.8417647
# F1 Micro: 0.9806818
# F1 Macro: 0.9807996
# F1 Weighted: 0.9808141
# PCA Explained Variance Ratio: 0.1673027


# Durations. We only have durations for the last runs, because we found an error on the estimation of the duration.
word_mlp <- df[df$tokenizer == "short-word" & df$classifier == "mlp", ]
word_rf <- df[df$tokenizer == "short-word" & df$classifier == "rf", ]
bigram_mlp <- df[df$tokenizer == "bigram" & df$classifier == "mlp", ]
bigram_rf <- df[df$tokenizer == "bigram" & df$classifier == "rf", ]
plot(duration ~ voc_size, data = word_mlp, col = "red", ylim = c(0, 600), xlab = "Vocabulary Size", ylab = "Duration (s)")
points(duration ~ voc_size, data = word_rf, col = "green")
points(duration ~ voc_size, data = bigram_rf, col = "blue")
points(duration ~ voc_size, data = bigram_mlp, col = "purple")
legend(500, 600, legend = c("MLP - Short-Word", "MLP - Bigram", "RF - Short-Word", "RF - Bigram"), col = c("red", "purple", "green", "blue"), lty = 1)
