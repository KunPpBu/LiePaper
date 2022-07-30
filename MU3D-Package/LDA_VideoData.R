library(MASS)
#import video level dataset
MU3D_Video_Level_Data0 <- read.csv("MU3D_Video_Level_Data.csv")
MU3D_Video_Level_Data <- MU3D_Video_Level_Data0[,-c(1,14)]
#SVM split train and test 80/20
smp_size_raw <- floor(0.80 * nrow(MU3D_Video_Level_Data))
train_ind_raw <- sample(nrow(MU3D_Video_Level_Data), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_Video_Level_Data[train_ind_raw, ])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data[-train_ind_raw, ])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)


# feature extraction
set.seed(3117)
rfeCNTL <- rfeControl(functions = ldaFuncs, method = "cv", number = 11)
lda.features <- rfe(train_raw.df[,c(1,3:12)], train_raw.df[,2],
                    sizes = c(12:3),
                    rfeControl = rfeCNTL)

lda.features
colnames(lda.features$fit$means) #16 features



id <- match(colnames(lda.features$fit$means), colnames(MU3D_Video_Level_Data))
train_raw.df <- as.data.frame(MU3D_Video_Level_Data[train_ind_raw, c(id,2)])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data[-train_ind_raw, c(id,2)])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)

#LDA 
lda.fit <- lda(Veracity~., data = train_raw.df)
lda.fit

plot(lda.fit, type ="both")

lda.predict <- predict(object = lda.fit, newdata = test_raw.df)
head(lda.predict$posterior)
head(lda.predict$x)

#find accuracy of model
mean(lda.predict$class==test_raw.df$Veracity)
#64%


#create plot
LD1_proj <- lda.predict$x
class = test_raw.df$Veracity
df  <- data.frame(LD1 = LD1_proj, class = as.factor(class))

ggplot(data = df)+
  geom_density(aes(LD1, fill = class), alpha = 0.1)



