library(readr)
library(FactoMineR)
library(mice)
library(ggplot2)
library(dplyr)

raw <- read.csv("data/dataset.csv")
summary(raw)


# Inspect language distribution
language_distribution <- table(raw$language)
print(language_distribution)

# Check proportions
language_proportions <- prop.table(language_distribution)
print(language_proportions)

barplot(language_distribution, main="Language Distribution", col="lightblue", las=2)

cat("Dominant languages:\n")
print(sort(language_proportions, decreasing = TRUE)[1:5])

cat("\nMinority languages:\n")
print(sort(language_proportions, decreasing = TRUE)[length(language_proportions)-5:length(language_proportions)])
