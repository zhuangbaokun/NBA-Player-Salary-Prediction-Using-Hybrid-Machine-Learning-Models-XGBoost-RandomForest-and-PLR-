# Steps:
# 1) Clean Data (Done)
# 2) Check if Assumptions are met
# i) Correlation Matrix (Done)
# ii) VIF (Done)
# iii) Outliers ()
# 3) Feature Select (Continuous and Nominal Data) (Done Using RFE)
# 4) Model Building
# 5) Improvement of Model Through setting breakpoints (young, peak, old)
# 6) Feature Importance
# 7) Analysis

#### Import Data with no Rookies ####
library(readr)
library(dplyr)
library(tidyverse)
setwd("C:/Users/User/Desktop/ST4248 Term Paper")
df <- read.csv("C:/Users/User/Desktop/ST4248 Term Paper/df_cleaned.csv")
withrookies <- read.csv("C:/Users/User/Desktop/ST4248 Term Paper/withrookies.csv")
df = df[-c(1)]
df[,c(38,2)]%>%glimpse()
# Continuous vs. Nominal: run an ANOVA. In R, you can use ?aov.
# Nominal vs. Nominal: run a chi-squared test. In R, you use ?chisq.test.


##### Boxplot ######
library(ggplot2)
ggplot(df,aes(group=rn,x=AGE,y=SALARY_MILLIONS))+geom_boxplot()
boxplot(SALARY_MILLIONS~AGE,
        data=df,
        main="Different boxplots for each Age",
        xlab="Age",
        ylab="Salary in Millions",
        col="orange",
        border="brown"
)
boxplot(SALARY_MILLIONS~AGE,
        data=withrookies,
        main="Different boxplots for each Age",
        xlab="Age",
        ylab="Salary in Millions",
        col="orange",
        border="brown"
)


###Correlation HeatMap###
df%>%info()
df%>%dim()
df%>%glimpse()
df%>%names()
# Set aside dfs###############################
cat = df[c(2,3,28)]
res = df[c(38)]
number = df[-c(2,3,28,38)]
player_and_team = df[c(2,28)]
#############################################
cormat <- round(cor(number),2)
library(reshape2)
melted_cormat <- melt(cormat)
library(ggplot2)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
upper_tri
# Melt the correlation matrix
library(reshape2)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Heatmap
library(ggplot2)
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)
# Possible Presence of Multicollinearity -> VIF



#### Variance Inflation Factor (VIF)#### to remove variables that have multicollinearity problems

library(car)
vif_stepwise=function(df){
  df1 = df[-c(2,3,28,32)]
  lst = c()
# Dropped DRPM as well due to aliases with ORPM
    while(vif(lm(SALARY_MILLIONS ~ ., df1))%>%max()>5){
    vif = vif(lm(SALARY_MILLIONS ~ ., df1))
    index = which.max(vif)
    colname = df1%>%names()
    print(colname[index])
    lst = c(lst,colname)
    df1 = df1[-c(index)]
    }
  return(df1)
}
# Column removed: "MP", "FGA","POINTS","TRB","FG","FT","X3P","X2P","FG.","MPG","WINS_RPM","Rk","ORPM","TOV","PIE"
new_df = vif_stepwise(df)
new_df%>%dim()


########Remove Highly Correlated Variables###############
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# calculate correlation matrix
correlationMatrix <- cor(new_df[,1:18])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.70)
# print indexes of highly correlated attributes
print(highlyCorrelated)
########## Remove and Check Again##################
# # calculate correlation matrix
# correlationMatrix <- cor(new_df[,1:18][,-c(highlyCorrelated)])
# # summarize the correlation matrix
# print(correlationMatrix)
# # find attributes that are highly corrected (ideally >0.75)
# highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.70)
# # print indexes of highly correlated attributes
# print(highlyCorrelated)
# new_df[,c(highlyCorrelated)]%>%names()
# Dropped columns : "FTA"  "DRB"  "GP"   "eFG."
new_df = new_df[,-c(highlyCorrelated)]
new_df = new_df%>%cbind(df[c(3)])
new_df%>%names()
new_df%>%dim()



########Feature Selection###########
# Feature Selection Using Recursive Feature Selection (Suitable for Nominal and continous data)
# https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
set.seed(123)
# Train_test_split
train=sample(1:nrow(new_df), floor(nrow(new_df)*0.70))
# load the library
library(mlbench)
library(caret)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(new_df[train,-c(15)], new_df[train,15], sizes=c(1:15), rfeControl=control, metric = "MSE")
# summarize the results
print(results)
# list the chosen features
to_keep = predictors(results)
# plot the results
plot(results, type=c("g", "o"))
# c("X2PA", "AGE",  "PF",   "FT.",  "X3PA", "RPM",  "BLK",  "ORB",  "STL" )
data = new_df%>%select(c("X2PA", "AGE",  "PF",   "FT.",  "X3PA", "RPM",  "BLK",  "ORB",  "STL","SALARY_MILLIONS"))




############## Model Fitting ########################
# Linear Regression
# RandomForest
# Gradient Boosting Algorithm


data%>%names()
######## Without Setting Cut-Points #################
get_mse = function(model,data,train,to_remove){
  return(mean((predict(model,data[-train,-to_remove])-data[-train,c(10)])**2))
}

# Linear Regresion
# set.seed(123)
# lr0 <- lm(SALARY_MILLIONS ~ ., data=data[train,-c(3,4,6,7,9)])
# modelSummary <- summary(lr0) 
# summary(lr0)

set.seed(1)
lr <- lm(SALARY_MILLIONS ~ ., data=data[train,-c(3,4,6,7,9)])
modelSummary <- summary(lr) 
summary(lr)
modelCoeffs <- modelSummary$coefficients
get_mse(lr,data,-train,c(3,4,6,7,9))
get_mse(lr,data,train,c(3,4,6,7,9))
# c(3,4,6,7,9)

# Penalized LR

# Prediction Function
penalized_pred = function(model,data, train,to_remove){
  x.test <- model.matrix(SALARY_MILLIONS ~., data[-train,-to_remove])[,-1]
  predictions <- model %>% predict(x.test) %>% as.vector()
  # Model performance metrics
  return(mean((data[-train,]$SALARY_MILLIONS - predictions)**2))
  # df = data.frame(
  #   MSE = MSE(predictions, data[-train,]$SALARY_MILLIONS),
  #   Rsquare = R2(predictions, data[-train,]$SALARY_MILLIONS)
  # )
}

# Predictor variables
library(tidyverse)
library(caret)
library(glmnet)
x <- model.matrix(SALARY_MILLIONS ~ ., data=data[train,-c(3,4,6,7,9)])[,-1]
x0 <- model.matrix(SALARY_MILLIONS ~ ., data=data[train,])[,-1]
# Outcome variable
y <- data[train,]$SALARY_MILLIONS
# # Ridge Reg
# set.seed(123)
# cv <- cv.glmnet(x0, y, alpha = 0, nfolds=5) #10 is default
# rlr <- glmnet(x0, y, alpha = 0, lambda = cv$lambda.min)
# penalized_pred(rlr,data,train,c(1000)) #4.691866
# coef(rlr)

set.seed(123) 
cv <- cv.glmnet(x, y, alpha = 0, nfolds=5) #10 is default
rlr1 <- glmnet(x, y, alpha = 0, lambda = cv$lambda.min)
penalized_pred(rlr1,data,-train,c(3,4,6,7,9)) #train
penalized_pred(rlr1,data,train,c(3,4,6,7,9)) #test
# > penalized_pred(rlr1,data,-train,c(3,4,6,7,9))
# [1] 20.64466
# > penalized_pred(rlr1,data,train,c(3,4,6,7,9))
# [1] 28.4188



# # Lasso
# set.seed(123)
# cv <- cv.glmnet(x0, y, alpha = 1, nfolds=5) #10 is default
# llr <- glmnet(x0, y, alpha = 1, lambda = cv$lambda.min)
# penalized_pred(llr,data,train,c(1000))

# 
# set.seed(123)
# cv <- cv.glmnet(x, y, alpha = 1, nfolds=5) #10 is default
# llr1 <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)
# penalized_pred(llr1,data,train,c(3,4,6,7,9))
# coef(llr1)


# RandomForest
library(randomForest)
library(mlbench)
library(caret)


# Random Search
mse = function(model,data,index){
  return(mean((data[index,]$SALARY_MILLIONS-predict(model,data[index,]))**2))
}
control <- trainControl(method="repeatedcv", number=5, repeats=3, search="random")
set.seed(123)
mtry <- floor(ncol(new_df)/3)
rf_random <- train(SALARY_MILLIONS~., data=data, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)
optimal_mtry = 8
rf = randomForest(SALARY_MILLIONS ~.,data=data[train,], mtry=8,importance=TRUE)
importance(rf)
varImpPlot(rf)
plot(data$X2PA,data$SALARY_MILLIONS)

mse(rf,data,train)
mse(rf,data,-train)
data%>%names()
# > mse(rf,data,train)
# [1] 3.872333
# > mse(rf,data,-train)
# [1] 25.7988



###Neural Network####
# control <- trainControl(method="repeatedcv", number=5, repeats = 3, verbose = TRUE, search = "grid")
# metric <- "RMSE"
# # manual tuning of Hyper parameters
# grid <- expand.grid(size=c(seq(from = 1, to = 50, by = 10)),
#                     decay=c(seq(from = 0.0, to = 0.3, by = 0.1)))
# # Training process of Neural Network (nnet)
# fit.nnet <- caret::train(SALARY_MILLIONS~., data=data, method="nnet", 
#                          preProcess = c('center', 'scale','pca'), 
#                          metric="RMSE", trControl=control,
#                          tuneGrid = grid)
# print(fit.nnet)
# plot(fit.nnet)

# Gradient Algo
# XGBOOST
# Fit the model on the training set
set.seed(123)
model <- train(
  SALARY_MILLIONS ~., data = data[train,], method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
zS
model$bestTune
# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 22      50         2 0.3     0              0.6                1      0.75

xgbpredictions <- model %>% predict(data[-train,])
mse(model, data, train) #XGBoost
mse(model, data, -train) #XGBoost


########################### Do by StepSize############################################################
# Split training and test set into age
dtrain = data[train,]
dtrain$AGE%>%unique()
train1 = dtrain[dtrain$AGE<=22,]
test1 = dtest[dtest$AGE<=22,]
train2 = dtrain[(dtrain$AGE>22)&(dtrain$AGE<33),]
test2 = dtest[(dtest$AGE>22)&(dtest$AGE<33),]
train3 = dtrain[dtrain$AGE>=33,]
test3 = dtest[dtest$AGE>=33,]
# Useful Pred function
pr = function(df){
  return(mean((df$SALARY_MILLIONS - df$s0)**2))
}


#######################Ridge Regression with stepsize################################################ 
set.seed(123)
x1 <- model.matrix(SALARY_MILLIONS ~ ., data=train1)[,-1]
y1 <- train1$SALARY_MILLIONS
cv1 <- cv.glmnet(x1, y1, alpha = 0, nfolds=5) #10 is default
rlr1 <- glmnet(x1, y1, alpha = 0, lambda = cv1$lambda.min)

x2 <- model.matrix(SALARY_MILLIONS ~ ., data=train2)[,-1]
y2 <- train2$SALARY_MILLIONS
cv2 <- cv.glmnet(x2, y2, alpha = 0, nfolds=5) #10 is default
rlr2 <- glmnet(x2, y2, alpha = 0, lambda = cv2$lambda.min)

x3 <- model.matrix(SALARY_MILLIONS ~ ., data=train3)[,-1]
y3 <- train3$SALARY_MILLIONS
cv3 <- cv.glmnet(x3, y3, alpha = 0, nfolds=5) #10 is default
rlr3 <- glmnet(x3, y3, alpha = 0, lambda = cv3$lambda.min)
# Train
dft1 = cbind(train1,predict(rlr1,model.matrix(SALARY_MILLIONS ~., train1)[,-1]))
dft2 = cbind(train2,predict(rlr2,model.matrix(SALARY_MILLIONS ~., train2)[,-1]))
dft3 = cbind(train3,predict(rlr3,model.matrix(SALARY_MILLIONS ~., train3)[,-1]))
pr(dft1)
pr(dft2)
pr(dft3)
# > pr(dft1)
# [1] 0.8026691
# > pr(dft2)
# [1] 19.78681
# > pr(dft3)
# [1] 14.10966

# Test
df1 = cbind(test1,predict(rlr1,model.matrix(SALARY_MILLIONS ~., test1)[,-1]))
df2 = cbind(test2,predict(rlr2,model.matrix(SALARY_MILLIONS ~., test2)[,-1]))
df3 = cbind(test3,predict(rlr3,model.matrix(SALARY_MILLIONS ~., test3)[,-1]))
pr(df1)
pr(df2)
pr(df3)
# > pr(df1)
# [1] 1.823118
# > pr(df2)
# [1] 19.62447
# > pr(df3)
# [1] 8.558341

# Train
tfinaldf = rbind(dft1,dft2,dft3)
mean((tfinaldf$SALARY_MILLIONS - tfinaldf$s0)**2)

# Test
finaldf = rbind(df1,df2,df3)
mean((finaldf$SALARY_MILLIONS - finaldf$s0)**2)
# [1] 14.88275


####################RandomForest With cutpoints#####################################
# Grid Search
seed = 123
control <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:9))
rf_gridsearch <- train(SALARY_MILLIONS~., data=train2, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
rf1=randomForest(SALARY_MILLIONS ~.,data=train1, mtry=8,importance=TRUE)
rf2=randomForest(SALARY_MILLIONS ~.,data=train2, mtry=3,importance=TRUE)
rf3=randomForest(SALARY_MILLIONS ~.,data=train2, mtry=7,importance=TRUE)


# Train
rdft1 = train1%>%mutate(s0 = predict(rf1,train1))
rdft2 = train2%>%mutate(s0 = predict(rf2,train2))
rdft3 = train3%>%mutate(s0 = predict(rf3,train3))
pr(rdft1)
pr(rdft2)
pr(rdft3)
rffinaldft = rbind(rdft1,rdft2,rdft3)
mean((rffinaldft$SALARY_MILLIONS - rffinaldft$s0)**2)#Train MSE


# Test
rdf1 = test1%>%mutate(s0 = predict(rf1,test1))
rdf2 = test2%>%mutate(s0 = predict(rf2,test2))
rdf3 = test3%>%mutate(s0 = predict(rf3,test3))
pr(rdf1)
pr(rdf2)
pr(rdf3)
rffinaldf = rbind(rdf1,rdf2,rdf3)
mean((rffinaldf$SALARY_MILLIONS - rffinaldf$s0)**2)#Test MSE

#####################XGBoost with Cutpoint######################################
set.seed(123)
x1 <- train(SALARY_MILLIONS ~., data = train1, method = "xgbTree",trControl = trainControl("cv", number = 10))
x2 <- train(SALARY_MILLIONS ~., data = train2, method = "xgbTree",trControl = trainControl("cv", number = 10))
x3 <- train(SALARY_MILLIONS ~., data = train3, method = "xgbTree",trControl = trainControl("cv", number = 10))

x1$bestTune
x2$bestTune
x3$bestTune


# Train
xdft1 = train1%>%mutate(s0 = predict(x1,train1))
xdft2 = train2%>%mutate(s0 = predict(x2,train2))
xdft3 = train3%>%mutate(s0 = predict(x3,train3))
pr(xdft1)
pr(xdft2)
pr(xdft3)
xfinaldft = rbind(xdft1,xdft2,xdft3)
mean((xfinaldft$SALARY_MILLIONS - xfinaldft$s0)**2) #Train MSE

# Test
xdf1 = test1%>%mutate(s0 = predict(x1,test1))
xdf2 = test2%>%mutate(s0 = predict(x2,test2))
xdf3 = test3%>%mutate(s0 = predict(x3,test3))
pr(xdf1)
pr(xdf2)
pr(xdf3)
xfinaldf = rbind(xdf1,xdf2,xdf3)
mean((xfinaldf$SALARY_MILLIONS - xfinaldf$s0)**2) #Test MSE





# Each Age Group Apply different Model
allfinaldf = rbind(xdf1,rdf2,xdf3)
mean((allfinaldf$SALARY_MILLIONS - allfinaldf$s0)**2)





################### Feature Importance##################
# set.seed(7)
# library(caret)
# # prepare training scheme
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
# # train the model
# model <- train(SALARY_MILLIONS~., data=data, method="lvq", preProcess="scale", trControl=control)
# # estimate variable importance
# importance <- varImp(model, scale=FALSE)
# # summarize importance
# print(importance)
# # plot importance
# plot(importance)
# roc_imp2 <- varImp(model, scale = FALSE)
plot(roc_imp2, top = 9)

library(randomForest)
importance(rf2)
varImpPlot(rf2)


#############Analysis of Overpaid and Underpaid Players##########################
df%>%head()
age1 = df[df$AGE<=22,]
age2 = df[(df$AGE<=32)&(df$AGE>=23),]
age3 = df[df$AGE>=33,]
col = train1%>%names()
x1 <- train(SALARY_MILLIONS ~., data = train1, method = "xgbTree",trControl = trainControl("cv", number = 10))
rf2=randomForest(SALARY_MILLIONS ~.,data=train2, mtry=2,importance=TRUE)
x3 <- train(SALARY_MILLIONS ~., data = train3, method = "xgbTree",trControl = trainControl("cv", number = 10))

f1 = age1%>%mutate(salary_prediction = predict(x1,age1%>%select(col)))
f2 = age2%>%mutate(salary_prediction = predict(rf2,age2%>%select(col)))
f3 = age3%>%mutate(salary_prediction = predict(x3,age3%>%select(col)))
dffinal = rbind(f1,f2,f3)%>%mutate(salary_diff = salary_prediction - SALARY_MILLIONS)
diff = dffinal[order(dffinal$salary_diff),]%>%select(c("PLAYER","AGE","SALARY_MILLIONS","salary_prediction","salary_diff"))
diff2 = dffinal[order(-dffinal$salary_diff),]%>%select(c("PLAYER","AGE","SALARY_MILLIONS","salary_prediction","salary_diff"))

rownames(diff) <- 1:nrow(diff)
diff%>%head(10)
rownames(diff2) <- 1:nrow(diff2)
diff2%>%head(25)
#############Analysis of Overpaid and Underpaid Players##########################




# Assumptions to look out for https://www.statisticssolutions.com/assumptions-of-linear-regression/ 
# Linear relationship
# Multivariate normality
# No or little multicollinearity
# No auto-correlation
# Homoscedasticity
# note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis requires at least 20 cases per independent variable in the analysis.


# Anova Test and Chi Square Test



