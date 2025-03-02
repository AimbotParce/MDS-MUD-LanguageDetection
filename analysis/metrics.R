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

# Coverage as a function of tokenization
plot(df$test_coverage ~ df$tokenizer)


#F1 score as a function of test / tokenization / classifier 
plot(df$f1_weighted ~ df$test_coverage)
plot(df$f1_weighted ~ df$tokenizer)
plot(df$f1_weighted ~ df$classifier)


# combined factor levels (tokenizer + classifier)
df <- df %>%
  mutate(group = interaction(tokenizer, classifier))
df <- df %>%
  mutate(group = reorder(group, as.numeric(tokenizer)))


# Plot F1 as a function of tokenizer + classifier
boxplot(f1_weighted ~ group, 
        data = df, 
        las = 2,                         # Rotate x-axis labels
        cex.axis = 0.7,
        main = "F1 Weighted by Tokenizer and Classifier",
        xlab = "Tokenizer + Classifier", 
        ylab = "F1 Weighted",
        col = rainbow(length(unique(df$group)))
 )


# Group by tokenization method
tokenizer_colors <- rainbow(length(unique(df$tokenizer)))
df$group_color <- tokenizer_colors[as.factor(df$tokenizer)]
boxplot(f1_weighted ~ group, 
        data = df, 
        las = 2,                         # Rotate x-axis labels
        cex.axis = 0.7,
        main = "F1 Weighted by Tokenizer and Classifier",
        xlab = "Tokenizer + Classifier", 
        ylab = "F1 Weighted",
        col = df$group_color  # Color based on tokenizer
)
