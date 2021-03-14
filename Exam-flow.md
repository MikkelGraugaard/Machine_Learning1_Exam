---
title: "Exam 2020 Machine Learning 1"
author: 'Mikkel Nymark Graugaard'
date: "12/12/2020"
output: pdf_document
---

*Loading the data*
```{r eval=TRUE, include=FALSE}
library("DataExplorer")
library(readxl)
library("caret")
library("dplyr")
library("corrplot")
library("ggplot2")
library("stats")
library("MASS")
library("caTools")
library("recipes")
library("leaps")
library("MAMI")
library("glmnet")
library("car")
library("glmulti")
library("rJava")
library("glmulti")

audit.copy <- read.csv("~/R-data/ML/Exam/audit2.csv")
audit <- audit.copy
attach(audit)
```


*1. Load audit2.csv. Standardize all predictors. Split the data set using the first*
*1162 observations for training and save the remaining observations in a test*
*data set. How large is the share of Risk assigned observations in each data set?*


**Making the parameters in accordance to their description**
```{r eval=TRUE}
audit[,c(3,6,7,9:11)] <- lapply(audit[,c(3,6,7,9:11)], as.factor)
audit[,c(1,2,4,5,8)] <- lapply(audit[,c(1,2,4,5,8)], as.numeric)
str(audit)
```
Now the parameters are in accordance to their description.



It doesn't make any sense to normalize the categories, as they are categories.


Using the scale function in R in order to standardize the variables.
```{r eval=TRUE}
audit.scale <- audit %>% mutate_at(c("Sector_score", "PARA_A", "PARA_B", "TOTAL", "Money_Value"), ~(scale(.) %>% as.vector))
colMeans(audit.scale[,c(1,2,4,5,8)])
```
Can see that the means of the continuous variables is now 0, meaning that the standardization worked
Which scales the data and centers the data. It scales each element by subtracting the mean and dividing by the sd.


*splitting the data* 
1162/1550 = 75% 

I partitioned the data at a 75% for the training data and 25% in the test data and I have chosen to stratisfi so that there are equal ratios of firms classified as 1 and 0 for risk in the the training and test data set.
```{r eval=TRUE}
prop.table(table(audit.scale$Risk))
set.seed(123, "L'Ecuyer")
Training <- createDataPartition(audit.scale$Risk, p = 0.75, list = FALSE)
train <- audit.scale[ Training, ]
test  <- audit.scale[-Training, ]

uniquetrain1 <- unique(train$History)
test <- test[test$History %in% uniquetrain1,]

#check
prop.table(table(train$Risk))
prop.table(table(test$Risk))
```
The split is 1163 in the training set and 387 in the test set. 

Here one can the the share of Risk 0 and 1 for training and test set. Since I stratisfied, then they are roughly the same. 


|           | 0              | 1            |
| :---       |    :----:   |          ---: |
| train     | 0.5674979       | 0.4325021    |
| test       | 0.5348837        | 0.4651163   |







*2. Estimate a model for predicting Risk using logistic regression and all available*
   *predictors. What is the training and test set accuracy? Interpret your result*


In logistic regression we use the maximum likelihood method to make the best fitting model. Maximum likelihood will provide values of β0 and β1 which maximize the probability of obtaining the data set.

```{r eval=TRUE}
logist.fit <- glm(Risk ~ ., family="binomial", data=train)
```



Here finding the training error.
This is done on the training data set.
```{r eval=TRUE}
logist.pred <- predict(logist.fit, newdata=train, type="response") #Making the predictions

cm_reg.train <- confusionMatrix(factor(ifelse(logist.pred > 0.5, "1", "0")), 
                          factor(train$Risk), positive = "1") #Classifying the predictions
cm_reg.train
```

Confusion Matrix and Statistics


|           | Reference 0    | Reference 1   |
| :---       |    :----:   |          ---: |
|Prediction 0    | 553       | 185      |
|Prediction 1     | 107      | 318       |


                                         
               Accuracy : 0.7489         
                 95% CI : (0.723, 0.7736)
    No Information Rate : 0.5675         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.4789         
                                         
     Mcnemar's Test P-Value : 6.603e-06      
                                         
            Sensitivity : 0.6322         
            Specificity : 0.8379         
         Pos Pred Value : 0.7482         
         Neg Pred Value : 0.7493         
             Prevalence : 0.4325         
         Detection Rate : 0.2734         
     Detection Prevalence : 0.3654         
      Balanced Accuracy : 0.7350         
                                         
       'Positive' Class : 1  


**Confusion Matrix:** Looking at the Confusion matrix, then the model predicts, 553 Risk 0 correctly and 318 Risk 1 correctly. The model also predicts 185 False 0 (prdicted 0 which were actually 1) and 107 false 1 (predicted as 1, but actually 0)

Accuracy = 0.7489 - This means that the model is predicting the right outcome on the test data 74,89% of the time, generally want the accuracy to be at least above 50%. I think 74,89% is fairly good. One would normally also be at least as good as classifying all to the most frequent class, here that would be Risk 0. If we had classified everything as Risk 0, then we would have had an accuracy of 56.74%

The 95% Confidence interval is between (0.723, 0.7736) - This means that we are 95% of the time the accuracy of our model will be between 72.3% and 77,36% when exposed to new unknown data.

Sensitivity = 0.6322 - This is how good our model is at predicting the actual 1. In this case our model will say that there is Risk 63.22% of the times where there actually is Risk. This is not great.

Specificity = 0.8379 - This is how good our model is at predicting the actual 0. In this case our model will say that there isn't Risk 83.79% of the times where there actually isn't any risk. This is great. It makes sense, as we have more observations with Risk 0, so it should be "easier" to predict 0. 

In this case I would prefer a model that is more "strict" and trade some of the specificity for sensitivity as I think it is more important to correctly classify firms with possible audit fraud. This could be done by lowering the threshold for classifying Risk 1, eg. 0,4.




*Here i do it on the test set.*
```{r eval=TRUE}
logist.pred2 <- predict(logist.fit, newdata=test, type="response") #Making the predictions

cm_reg <- confusionMatrix(factor(ifelse(logist.pred2 > 0.5, "1", "0")), 
                          factor(test$Risk), positive = "1")  #Classifying the predictions
cm_reg
```
Confusion Matrix and Statistics

|           | Reference 0    | Reference 1   |
| :---       |    :----:   |          ---: |
|Prediction 0    | 179      | 73      |
|Prediction 1     | 28     | 107       |

                                          
               Accuracy : 0.739           
                 95% CI : (0.6922, 0.7821)
    No Information Rate : 0.5349          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4668          
                                          
    Mcnemar's Test P-Value : 1.197e-05       
                                          
            Sensitivity : 0.5944          
            Specificity : 0.8647          
         Pos Pred Value : 0.7926          
         Neg Pred Value : 0.7103          
             Prevalence : 0.4651          
         Detection Rate : 0.2765          
      Detection Prevalence : 0.3488          
       Balanced Accuracy : 0.7296          
                                          
       'Positive' Class : 1  


The same way of interpreting this output.

*Train:*   
Accuracy :  0.7489  
Sensitivity : 0.6322         
Specificity : 0.8379 


*Test:*    
Accuracy : 0.739
Sensitivity : 0.5944          
Specificity : 0.8647 

It this is generally worse performing since it is done on the test set, a set which is unknown for the model. This is thus the correct way of testing the model in order to see how good it would be performing on unknown future data. 







*3. Estimate a model for predicting Risk using logistic regression and all available*
*predictors plus their two-way interactions. Calculate the in-sample error (loglikelihood* *function) and the AIC and BIC for this model and the model from*
*2). Which model is suggested by each of the three criteria? Interpret your*
*results thoroughly.*

This time in order to make interaction I do the the : sign, which creates the interaction between all variables. 

```{r eval=TRUE}
logist.fit.interact <- glm(Risk ~ . + .:., family="binomial", data=train)
```


Here finding the training error with two-way interactions.
This is done on the training data set.
```{r eval=TRUE}
logist.pred.interact <- predict(logist.fit.interact, newdata=train, type="response") #Making the predictions

cm_reg.train.interact <- confusionMatrix(factor(ifelse(logist.pred.interact > 0.5, "1", "0")), 
                          factor(train$Risk), positive = "1") #Classifying the predictions
cm_reg.train.interact
```
         
         
|           | Reference 0    | Reference 1   |
| :---       |    :----:   |          ---: |
|Prediction 0    | 476       | 120     |
|Prediction 1     | 184      | 383       |
                                          
               Accuracy : 0.7386          
                 95% CI : (0.7123, 0.7637)
    No Information Rate : 0.5675          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4754          
                                          
    Mcnemar's Test P-Value : 0.0003023       
                                          
            Sensitivity : 0.7614          
            Specificity : 0.7212          
         Pos Pred Value : 0.6755          
         Neg Pred Value : 0.7987          
             Prevalence : 0.4325          
         Detection Rate : 0.3293          
     Detection Prevalence : 0.4875          
        Balanced Accuracy : 0.7413          
                                          
       'Positive' Class : 1  



The in sample error is 1-0.7386 = 0.2614. For the interaction model.
The in sample error is 1-0.7489 = 0.2511. For the standard model without interactions. 

The model only with the normal parameters (scaled) has a better in-sample error, at 25.11%
Where the model with all two-way interactions has a in-sample error at 26.14%. 
This is the rate of classifying wrongly on the data it was trained on.
One wants to be as close to 0 for the error rate as possible. 
It looks like the two-way interactions didn't do good for the model, and the new parameters may just create noise in the model. 

One would have expected the two-way interactions to perform better on the training data. as
In-sample minimization favors complex models.
Therefor it is a good idea when picking a model to base it on AIC equivalent to picking the best-predicting model in large samples,  
with the low risk gives good risk property in theory, but not good model selection. As this model have a lower variance, but for some bias. 

```{r eval=TRUE}
summary(logist.fit)$aic
summary(logist.fit.interact)$aic
```
                            AIC
    logist.fit:             1285.915
    logist.fit.interact:    22236.54

Here it can be seen that the complex model with all two-way interactions gets a heavy penalty for all the extra predictors that are included in the model.  
And thus in terms of the AIC criteria one should pick the simpler model as well. 


Looking at the BIC.
BIC is better compared to AIC for finding the true model and has a more consistent model selection as it is stricter on the inclusion of parameters.
```{r eval=TRUE}
BIC(logist.fit)
BIC(logist.fit.interact)
```
                              BIC
  logist.fit:                1407.325
  logist.fit.interact:       23051

Here it can be seen that both models are penalized more heavily. One would again choose the simpler model.  









*4. Use a model averaging method suitable for logistic regression models using all*
*available predictors for predicting risk. Which model receives the majority of*
*the weight? Is this the true model?*


Model averaging methods for likelihood models, including logit such as BMA and smoothed AIC. I choose to use the BMA method. The advantage of model smoothing is that in general All models are approximations, and the models might contain information there is good for prediciton. By using model averaging one can combine the models and give them weights which can stabilize each other.


```{r eval=TRUE}
ma_bic <- mami(train, outcome = "Risk", model = "binomial", method = "MA.criterion", criterion = "BIC")
summary(ma_bic)
```
It uses 6 different parameters: Distrcit_Loss, Score_A, Score_MV, PARA_A, PARA_B and TOTAL

BMA is not robust to misspecification, i.e. designed to pick the ”true“ model, not a low risk model. It weighs the different models in accordance to the posterior probability that it is the true model 

BMA is not risk-optimal, as being risk optimal is being good at predicting the right in general, optimizing accuracy. Where the BMA is trying to pick the true model. 

The Bayesian moving average uses the Bayesian information criteria to find the optimal weight of all the different models. This is not a Risk-optimal criterion, here AIC would be better, as the AIC tries to include more predictors for better predictions. Where BIC tend to leave more predictors out, and thus be better at choosing consistent models and the more true model.  

One cannot say it is the true model, as we don't know what the true model would be, but using BMA gives a good approximation. 









*5. Extend the predictor set further by adding all squared predictors and two-way*
*interactions. Use a regularization approach for logistic regression that yields*
*sparse solutions to estimate a model for predicting Risk. Propose a method to*
*select your tuning parameter. Predict the Risk class using this approach. What*
*is the test and training accuracy? Briefly interpret your results*


Here I choose to make a blueprint of the data with squared predictors and two-way interactions. 
It is best to standardize parameters before applying regularization method, which was done in with the scale in the beginning.

```{r eval=TRUE}
# Writing and applying the recipe

recipedata <- audit.scale # Copying audit data so I have the original data set should I make a mistake in relation to the recipe
model_recipe <- recipe(Risk ~ ., data=recipedata)
# Writing the recipe steps

attach(recipedata)

model_recipe_steps <- model_recipe %>%
  step_poly(Sector_score, PARA_A, PARA_B, TOTAL, Money_Value
            ,degree = 2 #Second order polynomial
            ,options = list(raw = TRUE)
  ) %>% #Setting in second degree to square of all numeric predictors, as it doesn't make sense for categorical variables.
  step_dummy(all_nominal(), -Risk , one_hot = T) %>%  # All non-numeric i.e. categorical
  step_interact(terms = ~ all_predictors():all_predictors()) %>% # Doing the two-way interaction  
  step_zv(all_predictors()) # Removing zero variance parameters, as my regulazation didn't want to run before

detach(recipedata)

# Preparing
prepped_recipe <- prep(model_recipe_steps, data=recipedata)



# Splitting the data:
set.seed(123)
train_index_recipe <- createDataPartition(recipedata$Risk, p = .75, list = FALSE) # .075 split of the test and training data
recipe_train <- recipedata[ train_index_recipe,]
recipe_test  <- recipedata[-train_index_recipe,]

# Baking the recipe (applying recipe to your data sets)
baked_train <- bake(prepped_recipe, recipe_train)
baked_test <- bake(prepped_recipe, recipe_test)
```





I choose to to use the lasso regulazation in order to get sparse solution as it can set parameters to 0, as the ridge regression would not remove any parameters and are those not a sparse solution. 

I will be using 10 fold cross validation to find the best tuning parameter which is lambda

```{r eval=TRUE}
#auxiliary model and model.matrix for test
lm.test <- glm(Risk~., family="binomial", data=baked_test)
lm.train <- glm(Risk~., family="binomial", data=baked_train)

X <- model.matrix(lm.train)[,!is.na(lm.train$coefficients)][,-1]
x_test <- model.matrix(lm.test)[,!is.na(lm.train$coefficients)][,-1]


#10-fold using cv.glmnet to obtain best lambda along grid
set.seed(42069,"L'Ecuyer") 
y <- baked_train$Risk
y_test <- as.numeric(baked_test$Risk)

#setup grid for lambda
l.grid <- 10^seq(-2,3,length=100)


cv10.glmnet <-  cv.glmnet(X,y,alpha=1, nfolds = 10, lambda = l.grid, family="binomial", type.measure = "class")
# Has to specify the loss-function to class, for classificaiton. Alpha 1 for lasso method. nfolds=10 for 10 fold CV. lambda using the grid. best lambda chosen by the use of 10 fold Crossvalidation.  

bestlam <- cv10.glmnet$lambda.min
bestlam

```
The tuning parameter chosen by 10-fold cross validation is 0.01417474.
The lambda is small, which means that the penalizaion is relative small.
There is 21 parameters selected. can be seen with this code: coefficients(cv10.glmnet)



I will be doing the prediction on the unknown data, the test data. 
```{r eval=TRUE}
#obtain predictions for given lambda, named as bestlam
l1.pred.train <- predict(cv10.glmnet, s=bestlam, newx= X)
l1.pred.test <- predict(cv10.glmnet, s=bestlam, newx = x_test)


#predicting risk for training 
cm.l1.train<- confusionMatrix(factor(ifelse(l1.pred.train>0.5,"1","0")), #using 0.5 as cutoff level 
                         factor(baked_train$Risk), positive = "1") #defines the important category -- Risk=1


#predicting risk for test
cm.l1.test<- confusionMatrix(factor(ifelse(l1.pred.test>0.5,"1","0")), #using 0.5 as cutoff level 
                         factor(baked_test$Risk), positive = "1") #defines the important category -- Risk=1
cm.l1.train$overall[1]
cm.l1.test$overall[1]
```
Train: Accuracy : 0.7248495 
Test:  Accuracy : 0.7131783 


The Lasso is often better when the parameters in the original model, not all are true predictors for the response
Where Ridge is better when all the parameters are true for estimating the model. 


Regulazation methods is usually used in order to decrease high dimensionallity, which i shaving many parameters, in order to make a simpler model. 










*6. Use a super learner to combine at least two reasonable classification methods to*
*predict the Risk class. Which method receives the most weight? Calculate the*
*test error of the super learner. Hint: For binomial likelihood methods you can*
*specify the options: method = ”method.NNloglik” and family = binomial in SuperLearner().*

```{r eval=TRUE}
y <- as.numeric(baked_train$Risk)
ytest <- (baked_test$Risk)

x <- data.frame(train[,-1])
xtest <- data.frame(test[,-1])
```


Looking at possible models to include.
```{r eval=TRUE}
library(SuperLearner)
listWrappers()
```



I have choosen to do a glmnet model and a step AIC model for the super learner. 
```{r eval=TRUE}
SL.methods <- c('SL.glmnet','SL.stepAIC')

model.SL <- SuperLearner(y, x, family=binomial(), method = method.NNloglik, SL.library = SL.methods)
```



```{r eval=TRUE}
model.SL$coef
```

SL.glmnet_All     SL.stepAIC_All 
           0.5                0.5
           
Can be seen at the weights are equal at 0.5 and 0.5 for the 2 models chosen. Which means that they are equally good. 

```{r eval=TRUE}
predict.SL <- predict.SuperLearner(model.SL, newdata = xtest)
conv.preds <- factor(ifelse(predict.SL$pred>=0.5,"1","0"))
cm.SL <- confusionMatrix(conv.preds, factor(ytest))

cm.SL
```
Confusion Matrix and Statistics

         
|           | Reference 0    | Reference 1   |
| :---       |    :----:   |          ---: |
|Prediction 0    | 207       | 0       |
|Prediction 1     | 0      | 180       |
                                     
               Accuracy : 1          
                 95% CI : (0.9905, 1)
    No Information Rate : 0.5349     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
                                     
     Mcnemar's Test P-Value : NA         
                                     
            Sensitivity : 1.0000     
            Specificity : 1.0000     
         Pos Pred Value : 1.0000     
         Neg Pred Value : 1.0000     
             Prevalence : 0.5349     
         Detection Rate : 0.5349     
     Detection Prevalence : 0.5349     
        Balanced Accuracy : 1.0000     
                                     
       'Positive' Class : 0   
       

Something is wrong in terms of my SuperLearner model, but I don't know what it is. 

The super learner i smart as it can create models with good predictions, but one cant use it for inference. 












# Problem 2



*Clearing everything*
```{r eval=TRUE}
rm(list=ls())


library("DataExplorer")
library(readxl)
library("caret")
library("dplyr")
library("corrplot")
library("ggplot2")
library("stats")
library("MASS")
library("caTools")
library("recipes")
library("leaps")
library("MAMI")
library("glmnet")
library("car")
library("glmulti")


audit.copy <- read.csv("~/R-data/ML/Exam/audit2.csv")
audit <- audit.copy
attach(audit)

audit[,c(3,6,7,9:11)] <- lapply(audit[,c(3,6,7,9:11)], as.factor)
audit[,c(1,2,4,5,8)] <- lapply(audit[,c(1,2,4,5,8)], as.numeric)
```




*Load audit2.csv. Discretize all the continuous variables (High/Low) using their*
*median as a cutoff. Note: If you are not able to do this task, consider working*
*only with the discrete variables in the dataset.*



I will be setting high values as 1 and low values as 0 
```{r eval=TRUE}
audit$Sector_score <- ifelse(audit$Sector_score >= (median(audit$Sector_score)),"1","0" )
audit$PARA_A <- ifelse(audit$PARA_A >= (median(audit$PARA_A)),"1","0" )
audit$PARA_B <- ifelse(audit$PARA_B >= (median(audit$PARA_B)),"1","0" )
audit$TOTAL <- ifelse(audit$TOTAL >= (median(audit$TOTAL)),"1","0" )
audit$Money_Value <- ifelse(audit$Money_Value >= (median(audit$Money_Value)),"1","0" )

str(audit)

```
Looks like it worked, will also change the charecters to facors.  

```{r eval=TRUE}
audit[,c(1,2,4,5,8)] <- lapply(audit[,c(1,2,4,5,8)], as.factor)
str(audit)
```
Perfect. 






**2. Split the data using the 1162 observation for training and the rest of the**
**observations for testing. Consider the task of classifying the firms as risky**
**or non-risky using a Naive Bayes Classifier. How do you interpret a-priori**
**and conditional probabilities in the output? Evaluate the performance of the**
**model using Type-I error, Type II error, Sensitivity, Specificity, Accuracy, and**
**AUC for suspicious firm classification.**


Splitting the data again with the 75% ratio as this is equal to 1163 observations.
```{r eval=TRUE}
prop.table(table(audit$Risk))
set.seed(123, "L'Ecuyer")
Training <- createDataPartition(audit$Risk, p = 0.75, list = FALSE)
train <- audit[ Training, ]
test  <- audit[-Training, ]

uniquetrain1 <- unique(train$History)
test <- test[test$History %in% uniquetrain1,]
```


*How do you interpret a-priori and conditional probabilities in the output?*
The priori is the original probability of Risk 1 in the data set. 

```{r eval = TRUE}
prop.table(table(train$Risk))
```
The priori in the training set is 0.4436. 

       0         1   
0.5674979 0.4325021 


The conditional probability is the probability of being classified as 1 given another situation has happend. 


```{r eval=TRUE}
library(e1071)
nb <- naiveBayes(as.factor(Risk) ~ ., data = train)
nb

```
Example of The conditional probability

  
 |           | PARA_A 0    | PARA_A 1   |
| :---       |    :----:   |          ---: |
|Y 0    | 0.6196970      | 0.3803030     |
|Y 1     |  0.3399602      | 0.6600398       |

One can see that if a Firm is 1 for PARA_A, then the probability of being classified as a risky firm (Risk=1)
is 0.66 which is 66%



Create a confusion matrix
```{r eval=TRUE}
pred.prob.naive <- predict(nb, newdata = test[,-12], type="class")
Risk <- test$Risk
test$Risk <- as.factor(test$Risk)
confusionMatrix(pred.prob.naive, test$Risk, positive = "1")
```
Confusion Matrix and Statistics

|           | Reference 0    | Reference 1   |
| :---       |    :----:   |          ---: |
|Prediction 0    | 153      | 64    |
|Prediction 1     |  54      | 116     |   
        
                                          
               Accuracy : 0.6951          
                 95% CI : (0.6466, 0.7406)
    No Information Rate : 0.5349          
    P-Value [Acc > NIR] : 9.431e-11       
                                          
                  Kappa : 0.385           
                                          
    Mcnemar's Test P-Value : 0.4074          
                                          
            Sensitivity : 0.6444          
            Specificity : 0.7391          
         Pos Pred Value : 0.6824          
         Neg Pred Value : 0.7051          
             Prevalence : 0.4651          
         Detection Rate : 0.2997          
     Detection Prevalence : 0.4393          
       Balanced Accuracy : 0.6918          
                                          
       'Positive' Class : 1 




Evaluating the model performance   
Type 1 Error is classifying a firm as Risk 1, where it is actually Risk 0   
Type 2 Error is classifying a firm as Risk 0, where it is actually Risk 1   

Type 1 Error    : 54   
Type 2 Error    : 64    
Sensitivity     : 0.6444    
Specificity     : 0.7391    
Accuracy        : 0.6951     

Confusion Matrix: Looking at the Confusion matrix, then the model predicts, 153 Risk 0 correctly and 116 Risk 1 correctly. The model also predicts 64 False 0 (predicted 0 which were actually 1) and 54 false 1 (predicted as 1, but actually 0) - the latter type 1 and type 2 errors

Accuracy = 0.6951 - This means that the model is predicting the right outcome on the test data 69,51% of the time, generally want the accuracy to be at least above 50%. I think 69.51% is fairly good. One would normally also be at least as good as classifying all to the most frequent class, here that would be Risk 0. If we had classified everything as Risk 0, then we would have had an accuracy of 56,77%

Sensitivity = 0.6444 - This is how good our model is at predicting the actual 1. In this case our model will say that there is Risk 64.44% of the times where there actually is Risk. This is not great.

Specificity = 0.7391 - This is how good our model is at predicting the actual 0. In this case our model will say that there isn't Risk 83.85% of the times where there actually isn't any risk. This is great. It makes sense, as we have more observations with Risk 0, so it should be "easier" to predict 0. 

In this case I would prefer a model that is more "strict" and trade some of the specificity for sensitivity as I think it is more important to correctly classify firms with possible audit fraud. This could be done by lowering the threshold for classifying Risk 1, eg. to 0,4 or even lower.  


```{r eval=TRUE}
colAUC(as.integer(pred.prob.naive), test$Risk, plotROC = TRUE)
```

AUC : 0.6917874

The AUC is the area under the ROC curve.
This can be used to compare different models, and then selecting the best model in terms of ones preferences, e.g. accuracy, sensitivity. Looking at an ROC curve the one generally want to be in the upper left corner of the ROC plot, as this gives high sensitivity and specificity. 
The AUC gives the overall accuracy of the model, when tuning the probability of false alarm.
Can be seen as one increase the probability of false alarm, which is making type 1 error, then the sensitivity increases, which makes great sense, as this would be comparable to lowering ones threshold of classifying Risk 1. 





*3. Discuss to what extent the Naive Bayes outperforms alternative models (e.g. from problem 1) in terms of* *their accuracy in predicting the risk-class, and theoretically what are the advantages of using a Naive Bayes* *model?*


The Naive Bayes is good when there is a large amount of p (predictors), it finds same probability that record belongs to a class, Risk 0 or 1 in this case, given the predictor values. Where it looks at due to the given circumstances for the predictor values, what is the probability of the different classes, and then it classifies to the majority class, the class with the biggest probability. 
Naive Bayes is good as it is not so computational heavy, which makes it fast and can handle many predictors. 
This is the product of the probabilities, divided by probability by the class and probability of all the variables 







*4. Estimate the probability of fraud for a firm with a high risk score value of the target-unit from summary report A and a high risk score value of the target-unit from summary report B.1. Using the Naive Bayes model* *with all the predictors.*
*Evaluate which combination of predictors and their corresponding values is associated with the highest* *probability of fraud. For these inferences, use the train dataset.* 


If being Score_A = 0.6 and Score_B.1 = 0.6 finding probability of risk 1 


A-priori probabilities:
Y
          0         1 
  0.5674979 0.4325021 


   Score_A
Y         0.2       0.4       0.6
  0 0.6439394 0.1969697 0.1590909
  1 0.3717694 0.1769384 0.4512922
  
  
   Score_B.1
Y          0.2        0.4        0.6
  0 0.95454545 0.03333333 0.01212121
  1 0.86083499 0.09343936 0.04572565
  

  
Firstly calculating probability of Risk 0
```{r eval=TRUE}
prob.0 <- 0.5674979 * 0.1590909 * 0.01212121 
prob.0
```
[1] 0.001094348


Secondly calculating probability of Risk 1
```{r eval=TRUE}
prob.1 <- 0.4325021 * 0.4512922 * 0.04572565 
prob.1
```
[1] 0.008924953

Then classifying in accordance to the on with highest probability. 
```{r eval=TRUE}
ifelse(prob.0>prob.1,"0","1")
```
Here we would classify the firm as being risky (Risk = 1)
```{r eval=TRUE}
prob.1/(prob.1+prob.0)
```
[1] 0.890776
It gives a probability of 89%



Evaluating the combination giving the highest probability of being classified as Risk 1

```{r eval=TRUE}
pred.prob.naive2 <- predict(nb, newdata = train, type = "raw")
predicted.naive2 <- cbind(train[,-12], pred.prob.naive2)
predicted.naive2
max(predicted.naive2[13])
```
The combination giving the highest probability, gives an probability of 0.999768 to be classified as Risk 1



```{r eval=TRUE}
which(predicted.naive2[13]>=0.999768)
pred.prob.naive2[617,]
```

I don't know how to find this combination.

















