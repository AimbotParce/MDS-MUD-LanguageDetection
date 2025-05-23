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



png(filename = "plots/f1_weighted_vs_diacritics_removal.png", width = 800, height = 600, res = 100)
boxplot(f1_weighted ~ remove_diacritics,
    data = df,
    main = "F1 Weighted by Diacritics Removal",
    ylab = "F1 Weighted",
    xlab = "Remove Diacritics",
    col = c("skyblue", "lightgreen")
)
dev.off()
wilcox.test(f1_weighted ~ remove_diacritics, data = df)

png(filename = "plots/f1_weighted_vs_url_removal.png", width = 800, height = 600, res = 100)
boxplot(f1_weighted ~ remove_urls,
    data = df,
    main = "F1 Weighted by URL Removal",
    ylab = "F1 Weighted",
    xlab = "Remove URLs",
    col = c("salmon", "lightyellow")
)
dev.off()
wilcox.test(f1_weighted ~ remove_urls, data = df)

png(filename = "plots/f1_weighted_vs_symbols_removal.png", width = 800, height = 600, res = 100)
boxplot(f1_weighted ~ remove_symbols,
    data = df,
    main = "F1 Weighted by Symbols Removal",
    ylab = "F1 Weighted",
    xlab = "Remove Symbols",
    col = c("lightblue", "lavender")
)
dev.off()
wilcox.test(f1_weighted ~ remove_symbols, data = df)


png(filename = "plots/f1_weighted_vs_sentence_splitting.png", width = 800, height = 600, res = 100)
boxplot(f1_weighted ~ split_sentences,
    data = df,
    main = "F1 Weighted by Sentence Splitting",
    ylab = "F1 Weighted",
    xlab = "Split Sentences",
    col = c("lightgreen", "lightpink")
)
dev.off()
wilcox.test(f1_weighted ~ split_sentences, data = df)

png(filename = "plots/f1_weighted_vs_lowercase.png", width = 800, height = 600, res = 100)
boxplot(f1_weighted ~ lower,
    data = df,
    main = "F1 Weighted by Lowercasing",
    ylab = "F1 Weighted",
    xlab = "Lowercase",
    col = c("lightcoral", "lightblue")
)
dev.off()
wilcox.test(f1_weighted ~ lower, data = df)

png(filename = "plots/f1_weighted_vs_classifier.png", width = 800, height = 600, res = 100)
classifier <- factor(df$classifier,
    levels = c("dt", "knn", "lda", "lr", "mlp", "nb", "rf", "svm"),
    labels = c("DT", "KNN", "LDA", "LR", "MLP", "NB", "RF", "SVC")
)
boxplot(df$f1_weighted ~ classifier,
    main = "F1 Weighted by Classifier",
    ylab = "F1 Weighted",
    xlab = "Classifier",
    col = c("lightgreen", "lightpink", "lightblue", "lightcoral", "lightyellow", "lightcyan", "lightsalmon", "lightgray")
)
dev.off()
kruskal.test(f1_weighted ~ classifier, data = df)

png(filename = "plots/f1_weighted_vs_tokenizer.png", width = 800, height = 600, res = 100)
tokenizers <- factor(df$tokenizer, levels = c("short-word", "bigram", "char", "word"), labels = c("Hybrid", "Bigram", "Unigram", "Word"))
boxplot(df$f1_weighted ~ tokenizers,
    main = "F1 Weighted by Tokenizer",
    ylab = "F1 Weighted",
    xlab = "Tokenizer",
    col = c("lightgreen", "lightpink", "lightblue", "lightcoral")
)
dev.off()
kruskal.test(f1_weighted ~ tokenizer, data = df)


summary(df)
model <- glm(f1_weighted ~ remove_diacritics + remove_urls + remove_symbols + split_sentences + lower, data = df)
summary(model)

# Find the best combination of parameters, based on the highest F1 Weighted
df_2000 <- df[df$voc_size == 2000, ]
best_2000 <- df_2000[which.max(df_2000$f1_weighted), ]
best_2000
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

best <- df[which.max(df$f1_weighted), ]
best
# Tokenizer: bigram
# Classifier: mlp
# Remove Diacritics: False
# Remove URLs: False
# Remove Symbols: False
# Split Sentences: False
# Lower: False
# ---
# Train Coverage: 0.9158911
# Test Coverage: 0.9173231
# F1 Micro: 0.9811364
# F1 Macro: 0.9812277
# F1 Weighted: 0.9812625
# PCA Explained Variance Ratio: 0.1613334

# Durations. We only have durations for the last runs, because we found an error on the estimation of the duration.
word_mlp <- df[df$tokenizer == "short-word" & df$classifier == "mlp", ]
word_rf <- df[df$tokenizer == "short-word" & df$classifier == "rf", ]
bigram_mlp <- df[df$tokenizer == "bigram" & df$classifier == "mlp", ]
bigram_rf <- df[df$tokenizer == "bigram" & df$classifier == "rf", ]
png(filename = "plots/duration_vs_vocabulary_size_granular.png", width = 800, height = 600, res = 100)
plot(duration ~ voc_size, data = word_mlp, col = "red", ylim = c(0, 600), xlab = "Vocabulary Size", ylab = "Duration (s)")
points(duration ~ voc_size, data = word_rf, col = "green")
points(duration ~ voc_size, data = bigram_rf, col = "blue")
points(duration ~ voc_size, data = bigram_mlp, col = "purple")
legend(500, 600, legend = c("MLP - Short-Word", "MLP - Bigram", "RF - Short-Word", "RF - Bigram"), col = c("red", "purple", "green", "blue"), lty = 1)
dev.off()


mlp <- df[df$classifier == "mlp", ]
rf <- df[df$classifier == "rf", ]
png(filename = "plots/duration_vs_vocabulary_size.png", width = 800, height = 600, res = 100)
plot(duration ~ voc_size, data = mlp, col = "red", ylim = c(0, 600), xlab = "Vocabulary Size", ylab = "Duration (s)")
points(duration ~ voc_size, data = rf, col = "blue")
legend(500, 600, legend = c("MLP", "RF"), col = c("red", "blue"), lty = 1)
dev.off()
