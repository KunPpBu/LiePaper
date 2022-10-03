library(reshape2)
library(factoextra)
library(psych)
library(corrplot)
library(FactoMineR)
library(devtools)
install_github('sinhrks/ggfortify',force = TRUE)
library(ggfortify)
library(e1071)
library(caret)
library(randomForest)

#import video level dataset
MU3D_Video_Level_Data0 <- read.csv("MU3D_Video_Level_Data.csv")
#import target level dataset
MU3D_Target_Level_Data0 <- read.csv("MU3D_Target_Level_Data.csv")

#################################
# Boxplot for variables - video
#################################
#remove veractiy first
MU3D_Video_Level_Data <- MU3D_Video_Level_Data0[,-3]
str(MU3D_Video_Level_Data)
#scaled
colnames(MU3D_Video_Level_Data)

MU3D_Video_Level_Data.scaled <- data.frame(scale(MU3D_Video_Level_Data[,-c(1,13)]))

#level veractiy
levels0 <- unique(c(MU3D_Video_Level_Data0$Veracity, MU3D_Video_Level_Data0$Veracity))
#add veracity back
MU3D_Video_Level_Data.scaled$Veracity <- factor(MU3D_Video_Level_Data0$Veracity,levels = levels0)

#melt data for boxplot
MU3D_Video_Level_Data.scaled.melt <- melt(MU3D_Video_Level_Data.scaled, id.var = "Veracity")

#plot boxplot
ggplot(data = MU3D_Video_Level_Data.scaled.melt, aes(x=Veracity, y = value))+
  geom_boxplot() + 
  facet_wrap(~variable, ncol=4)



#################################
# Correlation analysis - video
#################################

panel.hist <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  his <- hist(x, plot = FALSE)
  breaks <- his$breaks
  nB <- length(breaks)
  y <- his$counts
  y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = rgb(0, 1, 1, alpha = 0.5), ...)
  lines(density(x), col = 2, lwd = 2) 
}



# Creating the scatter plot matrix- scaled
pairs(MU3D_Video_Level_Data.scaled,
      upper.panel = NULL,         
      diag.panel = panel.hist) 



#################################
# Feature Selection - video
#################################
# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Veracity~., data=MU3D_Video_Level_Data.scaled, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

features <- colnames(MU3D_Video_Level_Data.scaled[,importance$importance$X0>=0.51])


#################################
# Random Forest - video
#################################

#RF split train and test 80/20
set.seed(123)
smp_size_raw <- floor(0.80 * nrow(MU3D_Video_Level_Data.scaled))
train_ind_raw <- sample(nrow(MU3D_Video_Level_Data.scaled), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[train_ind_raw, importance$importance$X0>=0.51])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[-train_ind_raw, importance$importance$X0>=0.51])
train_raw.df$Veracity <- MU3D_Video_Level_Data.scaled[train_ind_raw, 12]
test_raw.df$Veracity <- MU3D_Video_Level_Data.scaled[-train_ind_raw, 12]
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)


set.seed(123)
rf.fit <- randomForest(Veracity~., data= MU3D_Video_Level_Data.scaled)
rf.fit
plot(rf.fit)
which.min(rf.fit$err.rate[,1])

set.seed(123)
mtry <- tuneRF(MU3D_Video_Level_Data.scaled[,-12],MU3D_Video_Level_Data.scaled$Veracity, ntreeTry=1000,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[,2] == min(mtry[,2]), 1]
print(mtry)
print(best.m) 


set.seed(123)
rf.fit1 <-randomForest(Veracity~.,data=train_raw.df[,-3], mtry=best.m, importance=TRUE,ntree=334)
print(rf.fit1)
#Evaluate variable importance
importance(rf.fit1)
varImpPlot(rf.fit1)


set.seed(123)
pred1=predict(rf.fit1, test_raw.df, type = "prob")
library(ROCR)
perf = prediction(pred1[,2], test_raw.df$Veracity)
# 1. Area under curve
auc = performance(perf, "auc")
1-auc@y.values[[1]]

# 2. True Positive and Negative Rate
pred3 = performance(perf, "fpr","tpr")
# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("topleft", c(paste0("AUC = ", round(1-auc@y.values[[1]],4))))


rf.pred <- predict(rf.fit1, test_raw.df)
confusionMatrix(rf.pred, test_raw.df$Veracity)


#################################
# SVM - video
#################################
poly.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                      kernel = "polynomial",
                      degree = c(2, 3, 4, 5, 6),
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(poly.tune) #best degree is 4,coef0 = 2,  misclassification rate no larger than 12.87%
best.poly <- poly.tune$best.model
poly.test <- predict(best.poly, newdata = test_raw.df)
table(poly.test, test_raw.df$Veracity)
confusionMatrix(poly.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # 0.8125 accuracy, kappa = 0.6261


#################################
# Ensemble Learning - video
#################################
library(caretEnsemble)
set.seed(100)

control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)

algorithms_to_use <- c( 'glm', 'knn', 'svmPoly','svmLinear', 'wsrf',  'gbm')


stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

stacking_summary<- summary(stacking_results)
save(stacking_summary, file = "stacking_summary.RData")

glm_cm<- confusionMatrix(stacked_models$glm$pred$pred, stacked_models$glm$pred$obs)
knn_cm<- confusionMatrix(stacked_models$knn$pred$pred, stacked_models$knn$pred$obs)
svmPoly_cm<- confusionMatrix(stacked_models$svmPoly$pred$pred, stacked_models$svmPoly$pred$obs)
svmLinear_cm<- confusionMatrix(stacked_models$svmLinear$pred$pred, stacked_models$svmLinear$pred$obs)
wsrf_cm<- confusionMatrix(stacked_models$wsrf$pred$pred, stacked_models$wsrf$pred$obs)
gbm_cm<- confusionMatrix(stacked_models$gbm$pred$pred, stacked_models$gbm$pred$obs)

data <- data.frame(matrix(c(0.7109, 0.7422, 0.7969, 0.7953, 0.9578, 0.9734, 
                            0.7344, 0.7562, 0.9938, 1.0000, 0.9625, 0.9812,
                            0.6875, 0.7281, 0.6000, 0.5906, 0.9531, 0.9656, 
                            0.4219, 0.4844, 0.5938, 0.5906, 0.9156, 0.9469),6,4))
row <- c("RF+GLM", "RF+KNNs", "RF+SVM.Poly", "RF+SVM.Linear", "RF+WSRF", "RF+GBM")
col <- c("Accuracy", "Sensitivity", "Specificity", "Kappa")
colnames(data) <- col
data$Method <- row
rownames(data) <- c(1:6)
data1<- data[,c(5,1,2,3,4)]

#write.csv(data, "ensemble_result.csv")
#plotting 
library(ggplot2)

# create a dataset
specie <- c(rep("sorgho" , 3) , rep("poacee" , 3) , rep("banana" , 3) , rep("triticum" , 3) )
condition <- rep(c("normal" , "stress" , "Nitrogen") , 4)
value <- abs(rnorm(12 , 0 , 15))
data1 <- data.frame(specie,condition,value)
library(tidyr)
data2 <- tidyr::gather(data1, key="Measurement", value="Value", 2:5)

# Grouped
ggplot(data2, aes(fill=Measurement, y=Value, x=reorder(Method, -Value))) + 
  geom_bar(position="dodge", stat="identity")+
  xlab("Method")

# stack using rf
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(100)

glm_stack <- caretStack(stacked_models, method="rf", metric="Accuracy", trControl=stackControl)

print(glm_stack)


