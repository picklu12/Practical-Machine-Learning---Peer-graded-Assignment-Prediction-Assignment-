# Practical-Machine-Learning---Peer-graded-Assignment-Prediction-Assignment-
Practical Machine Learning - Peer-graded Assignment: Prediction Assignment 
---
title: 'Peer-graded Assignment: Prediction Assignment Writeup'
author: "Raju Roy"
date: "November 21, 2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown

##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Loading and Processing the data

```{r warning=FALSE,cache=TRUE}

if(!file.exists("./pml-training.csv")){
        file.create("./pml-training.csv")
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                                                            "./pml-training.csv")
}

if(!file.exists("./pml-testing.csv")){
        file.create("./pml-testing.csv")
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                                                            "./pml-testing.csv")
}

training_data<-read.csv("./pml-training.csv",na.strings = c("NA", "#DIV/0!", ""))
testing_data <- read.csv("./pml-testing.csv",na.strings = c("NA", "#DIV/0!", ""))
dim(training_data)
dim(testing_data)
```


###Cleaning the data and removing the irrebalent columns

```{r warning=FALSE,cache=TRUE}
training_dada <- training_data[, which(colSums(is.na(training_data)) == 0)] 
testing_data <- testing_data[, which(colSums(is.na(testing_data)) == 0)]
training_data<-training_data[,colSums(is.na(training_data)) == 0]
testing_data <-testing_data[,colSums(is.na(testing_data)) == 0]
training_data <- training_data[,-c(1:7)] 
testing_data <- testing_data[,-c(1:7)]
```
### Verify data cleaning is done properly
```{r}
sum(!complete.cases(training_data))
sum(!complete.cases(testing_data))
```
###Partioning the training set into training and tesing(validation) datasets
```{r warning=FALSE, cache=TRUE}
library(ggplot2)
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(gbm)
library(MASS)

set.seed(123)
training_data = data.frame(training_data)
inTrain <- createDataPartition(training_data$classe, p=0.80, list=F)
train <- training_data[inTrain, ]
validation <- training_data[-inTrain, ]
```

##Building model
I will use three methods, random forest, boosted trees and  multinormial logistic regression analysis to build models and select the best model

###Modelling with Random Forest
```{r warning=FALSE, cache=TRUE}
model1 <- randomForest(classe ~ ., data=train,method="class")
prediction1 <- predict(model1,validation)
confusionMatrix(validation$classe, prediction1)
```

###Modelling with boosted trees ("gbm")
```{r warning=FALSE, cache=TRUE}
model2 <- train(classe ~ ., method="gbm", data=train,trControl=trainControl(method = "repeatedcv", number = 5, repeats = 1),verbose=FALSE)
prediction2 <- predict(model2, validation)
confusionMatrix(validation$classe, prediction2)
```

```{r, echo = FALSE}
library(ggplot2)
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(gbm)
library(MASS)
```

### modeling with multinormial logistic regression
```{r}
model3 <- multinom(classe ~ ., data=train, maxit =500, trace=T)
prediction3 <- predict(model3,validation)
confusionMatrix(validation$classe, prediction3)
```
###Final Model selection
From above, I found that Random Forest model achieve the perfect score with highest accuracy. So I would  pick Random Forest to be our model. I would use the random forest model for predicting test samples.

##Prediction using testing dataset
I used the random forest model to predict the classe for the 20 test cases.
```{r warning=FALSE, cache=TRUE}
prediction <- predict(model1, newdata=testing_data)
prediction
```
