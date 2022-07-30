library(factoextra)
library(psych)
library(corrplot)
library(FactoMineR)
library(devtools)
install_github('sinhrks/ggfortify')
library(ggfortify)

#import video level dataset
MU3D_Video_Level_Data0 <- read.csv("MU3D_Video_Level_Data.csv")
#remove chr col
MU3D_Video_Level_Data <- MU3D_Video_Level_Data0[,-c(1,14)]
MU3D_Video_Level_Data$Veracity <-factor(MU3D_Video_Level_Data$Veracity)
str(MU3D_Video_Level_Data)

#fit PCA model
pca.fit <- prcomp(MU3D_Video_Level_Data[,-2], center = TRUE, scale. = TRUE)
summary(pca.fit)

#screeplot
screeplot(pca.fit, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

#Cumulative variance plot
cumpro <- cumsum(pca.fit$sdev^2 / sum(pca.fit$sdev^2))
plot(cumpro[0:12], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 5, col="blue", lty=5)
abline(h = 0.75357, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC5"),
       col=c("blue"), lty=5, cex=0.6)


fviz_pca_ind(pca.fit, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = MU3D_Video_Level_Data$Veracity, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Veracity") +
  ggtitle("2D PCA-plot from 12 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))


#fit LDA with the first 5 PCs
smp_size_raw <- floor(0.80 * nrow(MU3D_Video_Level_Data))
train_ind_raw <- sample(nrow(MU3D_Video_Level_Data), size = smp_size_raw)
train_raw.df <- as.data.frame(MU3D_Video_Level_Data[train_ind_raw, ])
test_raw.df <- as.data.frame(MU3D_Video_Level_Data[-train_ind_raw, ])
levels <- unique(c(train_raw.df$Veracity, test_raw.df$Veracity))
test_raw.df$Veracity  <- factor(test_raw.df$Veracity, levels=levels)
train_raw.df$Veracity <- factor(train_raw.df$Veracity, levels=levels)

pcafit.lda <- lda(pca.fit$x[,1:5] , grouping=MU3D_Video_Level_Data[,2], data = train_raw.df)
plot(pcafit.lda,type="both")



