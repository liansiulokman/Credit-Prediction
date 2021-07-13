
rm(list = ls())

library(forecast)
library(readxl)
library(MASS)
library(randomForest)
library(dplyr)
library(DMwR)
library(gdata)

pacman::p_load("caret","partykit","ROCR","lift","rpart","e1071","glmnet","MASS","xgboost")

credit_data<- read_excel(file.choose())
application_data <-read_excel(file.choose())

#Feature Engineering
credit_data$SEX <- as.factor(credit_data$SEX)
credit_data$EDUCATION <- as.factor(credit_data$EDUCATION)
credit_data$MARRIAGE <- as.factor(credit_data$MARRIAGE)
application_data$SEX <- as.factor(application_data$SEX)
application_data$EDUCATION <- as.factor(application_data$EDUCATION)
application_data$MARRIAGE <- as.factor(application_data$MARRIAGE)

#Bill Amount Ratio
credit_data$bill_amt_ratio1 <- credit_data$BILL_AMT1 * credit_data$PAY_1
credit_data$bill_amt_ratio2 <- credit_data$BILL_AMT2 * credit_data$PAY_2
credit_data$bill_amt_ratio3 <- credit_data$BILL_AMT3 * credit_data$PAY_3
credit_data$bill_amt_ratio4 <- credit_data$BILL_AMT4 * credit_data$PAY_4
credit_data$bill_amt_ratio5 <- credit_data$BILL_AMT5 * credit_data$PAY_5
credit_data$bill_amt_ratio6 <- credit_data$BILL_AMT6 * credit_data$PAY_6
application_data$bill_amt_ratio1 <- application_data$BILL_AMT1 * application_data$PAY_1
application_data$bill_amt_ratio2 <- application_data$BILL_AMT2 * application_data$PAY_2
application_data$bill_amt_ratio3 <- application_data$BILL_AMT3 * application_data$PAY_3
application_data$bill_amt_ratio4 <- application_data$BILL_AMT4 * application_data$PAY_4
application_data$bill_amt_ratio5 <- application_data$BILL_AMT5 * application_data$PAY_5
application_data$bill_amt_ratio6 <- application_data$BILL_AMT6 * application_data$PAY_6

#Pay Delay Factor
for(i in c("PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")){
  col_name <- paste( i , "Delay", sep = "_", collapse = NULL)
  credit_data[[col_name]] <- ifelse(credit_data[[i]] %in% c(-2,-1,0), 0, credit_data[[i]])
  application_data[[col_name]] <- ifelse(application_data[[i]] %in% c(-2,-1,0), 0, application_data[[i]])
}

#Total Bill
credit_data$total_bill <- credit_data$BILL_AMT1 + credit_data$BILL_AMT2 + credit_data$BILL_AMT3 + credit_data$BILL_AMT4 + credit_data$BILL_AMT5 + credit_data$BILL_AMT6
application_data$total_bill <- application_data$BILL_AMT1 + application_data$BILL_AMT2 + application_data$BILL_AMT3 + application_data$BILL_AMT4 + application_data$BILL_AMT5 + application_data$BILL_AMT6

#High Educated Single
credit_data$high_educated_single <- ifelse(as.numeric(credit_data$EDUCATION)<=2 & credit_data$MARRIAGE==2,1,0)
application_data$high_educated_single <- ifelse(as.numeric(application_data$EDUCATION)<=2 & application_data$MARRIAGE==2,1,0)

#Grouping for education and marriage factor
credit_data$EDUCATION <- ifelse(credit_data$EDUCATION %in% c(4,5,6), 0, credit_data$EDUCATION)
credit_data$MARRIAGE <- ifelse(credit_data$MARRIAGE %in% c(3), 0, credit_data$MARRIAGE)
application_data$EDUCATION <- ifelse(application_data$EDUCATION %in% c(4,5,6), 0, application_data$EDUCATION)
application_data$MARRIAGE <- ifelse(application_data$MARRIAGE %in% c(3), 0, application_data$MARRIAGE)


#split train and test data (steps)
set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = credit_data$default_0,
                               p = 0.8, list = FALSE)
training <- credit_data[ inTrain,]
testing <- credit_data[ -inTrain,]


####XGBoost (steps)

credit_data_matrix <- model.matrix(default_0~ ., data = credit_data)[,-1]

x_train <- credit_data_matrix[inTrain,]
x_test <- credit_data_matrix[-inTrain,]

y_train <-training$default_0
y_test <-testing$default_0

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.01,       # hyperparameter: learning rate
                       max_depth = 3,  # hyperparameter: size of a tree in each boosting
                       nround=450,       # hyperparameter: number of boosting iterations
                       objective = "binary:logistic"
                       
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
xgb_confMatrix <- confusionMatrix(as.factor(ifelse(XGboost_prediction>0.2211,1,0)),as.factor(y_test),positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value



####
#### Calculating possible Gain & Loss ###
####

xgb_loss <-xgb_confMatrix$table[3] * (-5000)
xgb_loss

xgb_gain <-xgb_confMatrix$table[1] * (1500) 
xgb_gain

xgb_profit <- xgb_loss + xgb_gain

xgb_profit/4800




###### Prediction

testing <- application_data
training <- credit_data

testing$default_0 <- 0 #add dummy column to testing dataset

x_train <- model.matrix(default_0~.-ID, data = training)[,-1]
x_test <- model.matrix(default_0~.-ID, data = testing)[,-1]

y_train <-training$default_0

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.01,       # hyperparameter: learning rate
                       max_depth = 3,  # hyperparameter: size of a tree in each boosting
                       nround=450,       # hyperparameter: number of boosting iterations
                       objective = "binary:logistic"
                       
)

credit_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
prediction_output <- as.factor(ifelse(credit_prediction>0.2211,1,0))
