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



####################
# SVM 
####################
# tuning best svm model for linear kernel
linear.tune <- tune.svm(Veracity ~ ., data = train_raw.df,
                        kernel = "linear",
                        cost = c(0.001, 0.01, 0.1, 1, 5, 10))
summary(linear.tune) #best cost is 1, misclassification rate no larger than 25%

toc <- Sys.time()
best.linear <- linear.tune$best.model
linear.test <- predict(best.linear, newdata = test_raw.df)
table(linear.test, test_raw.df$Veracity)
confusionMatrix(linear.test, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) 
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])



####################
# RF 
####################
#fit 2 
toc <- Sys.time()
rf.fit2 <-randomForest(Veracity~.,data=train_raw.df)
print(rf.fit2)
rf.pred2 <- predict(rf.fit2, test_raw.df)
confusionMatrix(rf.pred2, test_raw.df$Veracity)
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])



####################
# KNN  
####################
library(class)
toc <- Sys.time()

target_category <- train_raw.df$Veracity
test_category <- test_raw.df$Veracity
k=sqrt(dim(MU3D_Video_Level_Data)[1])
##run knn function
knn.fit <- knn(train_raw.df,test_raw.df,cl=target_category,k=k)

##create confusion matrix
confusionMatrix(knn.fit, test_raw.df$Veracity, dnn = c("Prediction", "Reference")) 
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])




####################
# GLM 
####################
toc <- Sys.time()

##run glm function
glm.fit <- glm(Veracity~. ,family = binomial(link = "logit"), train_raw.df,)
outcome <- predict(glm.fit, newdata = test_raw.df, type = 'response')
outcome1 <- as.factor(ifelse(outcome > 0.5, 1, 0))
##create confusion matrix
confusionMatrix(data = outcome1, test_raw.df$Veracity, dnn = c("Prediction", "Reference"))
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])




####################
# WSRF 
####################
install.packages("wsrf")
library(wsrf)

target <- "Veracity"
ds <- MU3D_Video_Level_Data.scaled
vars <- names(ds)

if (sum(is.na(ds[vars]))) ds[vars] <- na.roughfix(ds[vars])
ds[target] <- as.factor(ds[[target]])
(tt <- table(ds[target]))
form <- as.formula(paste(target, "~ ."))

toc <- Sys.time()
model.wsrf.1 <- wsrf(form, data=train_raw.df, parallel=FALSE)
print(model.wsrf.1)
wdrf.fit <- predict(model.wsrf.1, newdata=test_raw.df, type="class")$class

##create confusion matrix
confusionMatrix(wdrf.fit, test_raw.df$Veracity, dnn = c("Prediction", "Reference"))
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])



####################
# GBM 
####################
install.packages("gbm")
library(gbm)
toc <- Sys.time()
fit.gbm <- gbm(Veracity~. , data= train_raw.df, 
               distribution = 'multinomial',
               cv.folds = 10,
               shrinkage = .01,
               n.minobsinnode = 10,
               n.trees = 200)

pred <- predict.gbm(object = fit.gbm,
                   newdata = test_raw.df,
                   n.trees = 200,
                   type = "response")

##create confusion matrix
pred.labels = colnames(pred)[apply(pred, 1, which.max)]
result = data.frame(test_raw.df$Veracity, pred.labels)
confusionMatrix(test_raw.df$Veracity, as.factor(pred.labels))
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])




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
#save(stacking_summary, file = "stacking_summary.RData")

glm_cm<- confusionMatrix(stacked_models$glm$pred$pred, stacked_models$glm$pred$obs)
knn_cm<- confusionMatrix(stacked_models$knn$pred$pred, stacked_models$knn$pred$obs)
svmPoly_cm<- confusionMatrix(stacked_models$svmPoly$pred$pred, stacked_models$svmPoly$pred$obs)
svmLinear_cm<- confusionMatrix(stacked_models$svmLinear$pred$pred, stacked_models$svmLinear$pred$obs)
wsrf_cm<- confusionMatrix(stacked_models$wsrf$pred$pred, stacked_models$wsrf$pred$obs)
gbm_cm<- confusionMatrix(stacked_models$gbm$pred$pred, stacked_models$gbm$pred$obs)



#################################
#RF+GLM - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','glm')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfglm_cm<- confusionMatrix(stacked_models$glm$pred$pred, stacked_models$glm$pred$obs)
rfglm_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])


#################################
#RF+KNNs - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','knn')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfknn_cm<- confusionMatrix(stacked_models$knn$pred$pred, stacked_models$knn$pred$obs)
rfknn_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])




#################################
#RF+svmPoly - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','svmPoly')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfsvmPoly_cm<- confusionMatrix(stacked_models$svmPoly$pred$pred, stacked_models$svmPoly$pred$obs)
rfsvmPoly_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])



#################################
#RF+svmPoly - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','svmLinear')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfsvmLinear_cm<- confusionMatrix(stacked_models$svmLinear$pred$pred, stacked_models$svmLinear$pred$obs)
rfsvmLinear_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])




#################################
#RF+wsrf - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','wsrf')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfwsrf_cm<- confusionMatrix(stacked_models$wsrf$pred$pred, stacked_models$wsrf$pred$obs)
rfwsrf_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])





#################################
#RF+gbm - video
#################################
toc <- Sys.time()
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithms_to_use <- c( 'rf','gbm')
stacked_models <- caretList(make.names(Veracity) ~., data=MU3D_Video_Level_Data.scaled, trControl=control_stacking, methodList=algorithms_to_use)
stacking_results <- resamples(stacked_models)
stacking_summary<- summary(stacking_results)

rfgbm_cm<- confusionMatrix(stacked_models$gbm$pred$pred, stacked_models$gbm$pred$obs)
rfgbm_cm
tic <- Sys.time()
print(difftime(tic, toc, units = "secs")[[1]])





#################################
# ROC 
#################################
library(ROCR)









