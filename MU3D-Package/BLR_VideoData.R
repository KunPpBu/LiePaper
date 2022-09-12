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

logit <- glm(Veracity~., family = binomial,data = train_raw.df)
summary(logit)

logit_2 <- stepAIC(logit)

#summary(logit_2)
logit$aic

logit_2$aic

train_raw.df$Predict <- ifelse(logit_2$fitted.values >0.5,"pos","neg")
# Confusion Matrix
mytable <- table(train_raw.df$Veracity,train_raw.df$Predict)
rownames(mytable) <- c("Obs. neg","Obs. pos")
colnames(mytable) <- c("Pred. neg","Pred. pos")
mytable

# accuracy
accuracy<- sum(diag(mytable))/sum(mytable)
accuracy


#Recall Or Sensitivity


# Sensitivity
Sensitivity <- (mytable[2, 2]/sum(mytable[2, ]))
Sensitivity


# Specificity
Specificity <- (mytable[1, 1]/sum(mytable[1, ]))
Specificity


#kappa 
prob.chance <- (sum(mytable[1,])/sum(mytable)) * (sum(mytable[,1])/sum(mytable))
prob.agree <- accuracy

kappa <- (prob.agree-prob.chance)/(1-prob.chance)
kappa






