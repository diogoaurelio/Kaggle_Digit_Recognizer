#LogisticRegression.R
library(readr)
library(proto)
library(ggplot2)

set.seed(155)
train <- data.frame(read_csv('./../train.csv'))
labels <- train[,1]
features <- train[,-1]


