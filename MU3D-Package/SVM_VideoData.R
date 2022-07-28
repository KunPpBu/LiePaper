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
  facet_wrap(~variable, ncol=2)





#SVM split train and test 80/20
smp_size_raw <- floor(0.80 * nrow(MU3D_Video_Level_Data.scaled))
train_ind_raw <- sample(nrow(MU3D_Video_Level_Data.scaled), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[train_ind_raw, ])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[-train_ind_raw, ])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)

# tuning best svm model for linear kernel
linear.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                        kernel = "linear",
                        cost = c(0.001, 0.01, 0.1, 1, 5, 10))
summary(linear.tune) #best cost is 1, misclassification rate no larger than 25%
best.linear <- linear.tune$best.model
linear.test <- predict(best.linear, newdata = test_raw.df)
table(linear.test, test_raw.df$Veracity)
confusionMatrix(linear.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # 76.6% accuracy, kappa = 0.54

# 
# # tuning best svm model for sigmoid kernel
# sigmoid.tune <- tune.svm(Veracity ~. ,data = train_raw.df,
#                          kernel = "sigmoid",
#                          gamma = c(0.1,0.5,1,2,3,4),
#                          coef0 = c(0.1,0.5,1,2,3,4))
# summary(sigmoid.tune)
# best.sigmoid <- sigmoid.tune$best.model
# sigmoid.test <- predict(best.sigmoid, test_raw.df)
# table(sigmoid.test,test_raw.df$Veracity)
# confusionMatrix(sigmoid.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # poor 64%, kappa= 0.27 drop this kernel



# tuning best svm model for polynomial kernel
poly.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                        kernel = "polynomial",
                        degree = c(2, 3, 4, 5, 6),
                        coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(poly.tune) #best degree is 3,coef0 = 3,  misclassification rate no larger than 17%( better than linear kernel)
best.poly <- poly.tune$best.model
poly.test <- predict(best.poly, newdata = test_raw.df)
table(poly.test, test_raw.df$Veracity)
confusionMatrix(poly.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # 78.12% accuracy, kappa = 0.56




# tuning best svm model for radial kernel
rad.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                      kernel = "radial",
                      gamma = c(0.1, 0.5, 1, 2, 3, 4))
summary(rad.tune) #best gamma = 0.1,  misclassification rate no larger than 19%( better than linear kernel, but not good as polynomial)
best.rad <- rad.tune$best.model
rad.test <- predict(best.rad, newdata = test_raw.df)
table(rad.test, test_raw.df$Veracity)
confusionMatrix(rad.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # 79% accuracy, kappa = 0.6 


# feature extraction
set.seed(3117)
rfeCNTL <- rfeControl(functions = lrFuncs, method = "cv", number = 11)
svm.features <- rfe(train_raw.df[,1:11], train_raw.df[,12],
                    sizes = c(11, 10, 9, 8, 7, 6, 5),
                    rfeControl = rfeCNTL,
                    method = "svmLinear")
svm.features
svm.features$fit$coefficients # Accuracy, TruthProp, VidLength_ms, VidLength_sec, Valence 



# use above 8 features to train polynomial svm

#SVM split train and test 80/20

train_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[train_ind_raw, c(1,4,5,7,8,12)])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data.scaled[-train_ind_raw, c(1,4,5,7,8,12)])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)



# tuning best svm model for polynomial kernel
poly.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                      kernel = "polynomial",
                      degree = c(2, 3, 4, 5, 6),
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(poly.tune) #best degree is 4,coef0 = 4,  misclassification rate no larger than 19%( better than linear kernel)
best.poly <- poly.tune$best.model
poly.test <- predict(best.poly, newdata = test_raw.df)
table(poly.test, test_raw.df$Veracity)
confusionMatrix(poly.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) # 95.31% accuracy, kappa = 0.90



