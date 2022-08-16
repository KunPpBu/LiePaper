library(reshape2)
library(factoextra)
library(psych)
library(corrplot)
library(FactoMineR)
library(devtools)
install_github('sinhrks/ggfortify')
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


#SVM split train and test 80/20
smp_size_raw <- floor(0.80 * nrow(MU3D_Video_Level_Data.scaled))
train_ind_raw <- sample(nrow(MU3D_Video_Level_Data.scaled), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[train_ind_raw, ])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[-train_ind_raw, ])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)



rf.fit <- randomForest(Veracity~., data= MU3D_Video_Level_Data.scaled)
rf.fit
plot(rf.fit)

mtry <- tuneRF(MU3D_Video_Level_Data.scaled[,-12],MU3D_Video_Level_Data.scaled$Veracity, ntreeTry=1000,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[,2] == min(mtry[,2]), 1]
print(mtry)
print(best.m) 


set.seed(123)
rf.fit1 <-randomForest(Veracity~.,data=train_raw.df, mtry=best.m, importance=TRUE,ntree=500)
print(rf.fit1)
#Evaluate variable importance
importance(rf.fit1)
varImpPlot(rf.fit1)



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

