
# input data
MU3D_Video_Level_Data <- read.csv("MU3D_Video_Level_Data.csv")
dim(MU3D_Video_Level_Data)
#320
head(MU3D_Video_Level_Data)
# divided into two group: lie and truth

lie_dat <- subset(MU3D_Video_Level_Data, MU3D_Video_Level_Data$Veracity==0)
dim(lie_dat)
truth_dat <- subset(MU3D_Video_Level_Data, MU3D_Video_Level_Data$Veracity==1)
dim(truth_dat)

plot(lie_dat[,9:13])

plot(truth_dat[9:13])

plot(MU3D_Video_Level_Data)
install.packages('mlbench')
install.packages('MASS')
install.packages('pROC')
library(mlbench)
library(MASS)
library(pROC)
library(dplyr)
library(tidyr)

MU3D_imageData_BF <- read.csv("MU3D_imageData_BF.csv", 
                              +     col_names = FALSE)
test <- separate(MU3D_imageData_BF2, c("x", "y"))
#remove comma and brackets
MU3D_imageData_BF1 <- sapply(MU3D_imageData_BF, gsub, pattern = ",", replacement= "")
MU3D_imageData_BF1 <- data.frame(MU3D_imageData_BF1)
MU3D_imageData_BF2 <- sapply(MU3D_imageData_BF1, gsub, pattern = "[()]", replacement= "")
MU3D_imageData_BF2 <- data.frame(MU3D_imageData_BF2)
View(MU3D_imageData_BF2)
#write.csv(MU3D_imageData_BF2, "MU3D_imageData_BF2.csv")
MU3D_imageData_BF2<- read.csv("MU3D_imageData_BF2.csv")
sapply(MU3D_imageData_BF2,mode)

View(MU3D_imageData_BF2)

MU3D_imageData_BF2 <- MU3D_imageData_BF2[,-1]
dim(MU3D_imageData_BF2)
label <- rep(c(1,0),40) #1=truth, 0=lie
length(label)
MU3D_imageData_BF2$label <- label

#Support Vector Machine
smp_size_raw <- floor(0.90 * nrow(MU3D_imageData_BF2))
train_ind_raw <- sample(nrow(MU3D_imageData_BF2), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_imageData_BF2[train_ind_raw, c(2:138)])
test_raw.df <- as.data.frame(MU3D_imageData_BF2[-train_ind_raw, c(2:138)])
library(e1071)
library(caret)

levels <- unique(c(train_raw.df$label, test_raw.df$label))
test_raw.df$label  <- factor(test_raw.df$label, levels=levels)
train_raw.df$label <- factor(train_raw.df$label, levels=levels)

svmfit <- svm(label ~ ., data = train_raw.df, kernel = "sigmoid", cost = 25, scale = TRUE)
pred.svm <- predict(svmfit, test_raw.df)
pred.svm          

confusionMatrix(pred.svm, test_raw.df$label, dnn = c("Prediction", "Reference"))


#Binary Logistic Regression
library(mlbench)
library(MASS)
library(pROC)


logit <- glm(label~., family = binomial, data = train_raw.df)
logit_2 <- stepAIC(logit)

train_raw.df$Predict <- ifelse(logit_2$fitted.values >0.5,"pos","neg")
mytable <- table(train_raw.df$label,train_raw.df$Predict)
rownames(mytable) <- c("Obs. neg","Obs. pos")
colnames(mytable) <- c("Pred. neg","Pred. pos")
mytable


# accuracy
accuracy<- sum(diag(mytable))/sum(mytable)
accuracy



# k-Nearest Neighbor
library(class)
smp_size_raw <- floor(0.90 * nrow(MU3D_imageData_BF2))
train_ind_raw <- sample(nrow(MU3D_imageData_BF2), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_imageData_BF2[train_ind_raw, c(2:138)])
test_raw.df <- as.data.frame(MU3D_imageData_BF2[-train_ind_raw, c(2:138)])
target_category <- train_raw.df$label
test_category <- test_raw.df$label 
k=sqrt(dim(MU3D_imageData_BF2)[1]) ##run knn function
pr <- knn(train_raw.df,test_raw.df,cl=target_category,k=5) ##create confusion matrix
tab <- table(pr,test_category)
##this function divides the correct predictions by total number of predictions that tell u
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100} 
accuracy(tab)



#head(mtcars)
boxplot(Accuracy ~ Veracity, data=MU3D_Video_Level_Data)
boxplot(Accuracy ~ Valence, data=MU3D_Video_Level_Data)
boxplot(Accuracy ~ Sex, data=MU3D_Video_Level_Data)
boxplot(Accuracy ~ Race, data=MU3D_Video_Level_Data)
boxplot(Accuracy ~ TruthProp, data=MU3D_Video_Level_Data)
boxplot(Accuracy ~ Attractive, data=MU3D_Video_Level_Data)


plot(Accuracy ~ TruthProp, data=MU3D_Video_Level_Data)
