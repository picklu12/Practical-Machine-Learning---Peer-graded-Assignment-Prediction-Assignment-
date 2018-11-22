# Practical-Machine-Learning---Peer-graded-Assignment-Prediction-Assignment-
Practical Machine Learning - Peer-graded Assignment: Prediction Assignment 
Peer-graded Assignment: Prediction Assignment Writeup
Raju Roy
November 21, 2018
R Markdown
Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
Loading and Processing the data
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
## [1] 19622   160
dim(testing_data)
## [1]  20 160
Cleaning the data and removing the irrebalent columns
training_dada <- training_data[, which(colSums(is.na(training_data)) == 0)] 
testing_data <- testing_data[, which(colSums(is.na(testing_data)) == 0)]
training_data<-training_data[,colSums(is.na(training_data)) == 0]
testing_data <-testing_data[,colSums(is.na(testing_data)) == 0]
training_data <- training_data[,-c(1:7)] 
testing_data <- testing_data[,-c(1:7)]
Verify data cleaning is done properly
sum(!complete.cases(training_data))
## [1] 0
sum(!complete.cases(testing_data))
## [1] 0
Partioning the training set into training and tesing(validation) datasets
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
Building model
I will use three methods, random forest, boosted trees and multinormial logistic regression analysis to build models and select the best model
Modelling with Random Forest
model1 <- randomForest(classe ~ ., data=train,method="class")
prediction1 <- predict(model1,validation)
confusionMatrix(validation$classe, prediction1)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    1    0    0    0
##          B    5  751    3    0    0
##          C    0    2  681    1    0
##          D    0    0    3  640    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9959          
##                  95% CI : (0.9934, 0.9977)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9948          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9955   0.9960   0.9913   0.9969   1.0000
## Specificity            0.9996   0.9975   0.9991   0.9991   0.9997
## Pos Pred Value         0.9991   0.9895   0.9956   0.9953   0.9986
## Neg Pred Value         0.9982   0.9991   0.9981   0.9994   1.0000
## Prevalence             0.2855   0.1922   0.1751   0.1637   0.1835
## Detection Rate         0.2842   0.1914   0.1736   0.1631   0.1835
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9976   0.9967   0.9952   0.9980   0.9998
Modelling with boosted trees (“gbm”)
model2 <- train(classe ~ ., method="gbm", data=train,trControl=trainControl(method = "repeatedcv", number = 5, repeats = 1),verbose=FALSE)
prediction2 <- predict(model2, validation)
confusionMatrix(validation$classe, prediction2)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1096    8    7    3    2
##          B   27  711   19    0    2
##          C    0   19  659    4    2
##          D    0    3   25  612    3
##          E    4    5    3    7  702
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9635          
##                  95% CI : (0.9572, 0.9692)
##     No Information Rate : 0.2873          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9539          
##  Mcnemar's Test P-Value : 6.703e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9725   0.9531   0.9243   0.9776   0.9873
## Specificity            0.9928   0.9849   0.9922   0.9906   0.9941
## Pos Pred Value         0.9821   0.9368   0.9635   0.9518   0.9736
## Neg Pred Value         0.9890   0.9889   0.9833   0.9957   0.9972
## Prevalence             0.2873   0.1902   0.1817   0.1596   0.1812
## Detection Rate         0.2794   0.1812   0.1680   0.1560   0.1789
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9827   0.9690   0.9582   0.9841   0.9907
## Loading required package: lattice
## randomForest 4.6-14
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## The following object is masked from 'package:ggplot2':
## 
##     margin
## Loaded gbm 2.1.4
modeling with multinormial logistic regression
model3 <- multinom(classe ~ ., data=train, maxit =500, trace=T)
## # weights:  270 (212 variable)
## initial  value 25266.565787 
## iter  10 value 20325.488269
## iter  20 value 18061.771063
## iter  30 value 16754.928765
## iter  40 value 15718.018692
## iter  50 value 15107.456003
## iter  60 value 14679.584885
## iter  70 value 14476.886165
## iter  80 value 14372.806051
## iter  90 value 14290.792395
## iter 100 value 14219.186121
## iter 110 value 14171.369573
## iter 120 value 14146.508557
## iter 130 value 14119.628944
## iter 140 value 14101.176260
## iter 150 value 14059.404739
## iter 160 value 13995.833803
## iter 170 value 13879.294129
## iter 180 value 12801.333075
## iter 190 value 11687.905267
## iter 200 value 11157.465559
## iter 210 value 10954.068139
## iter 220 value 10887.878724
## iter 230 value 10876.970985
## iter 240 value 10876.624647
## iter 250 value 10876.358841
## iter 260 value 10876.356310
## final  value 10876.355918 
## converged
prediction3 <- predict(model3,validation)
confusionMatrix(validation$classe, prediction3)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 974  34  56  44   8
##          B  91 501  66  23  78
##          C  66  45 505  42  26
##          D  46  24  73 468  32
##          E  28  87  54  53 499
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7512          
##                  95% CI : (0.7374, 0.7647)
##     No Information Rate : 0.3072          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6845          
##  Mcnemar's Test P-Value : 2.852e-10       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8083   0.7250   0.6698   0.7429   0.7760
## Specificity            0.9478   0.9202   0.9435   0.9469   0.9323
## Pos Pred Value         0.8728   0.6601   0.7383   0.7278   0.6921
## Neg Pred Value         0.9177   0.9399   0.9231   0.9506   0.9550
## Prevalence             0.3072   0.1761   0.1922   0.1606   0.1639
## Detection Rate         0.2483   0.1277   0.1287   0.1193   0.1272
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8780   0.8226   0.8066   0.8449   0.8542
Final Model selection
From above, I found that Random Forest model achieve the perfect score with highest accuracy. So I would pick Random Forest to be our model. I would use the random forest model for predicting test samples.
Prediction using testing dataset
I used the random forest model to predict the classe for the 20 test cases.
prediction <- predict(model1, newdata=testing_data)
prediction
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
