library(corrplot)
#import target level dataset
MU3D_Target_Level_Data0 <- read.csv("MU3D_Target_Level_Data.csv")
#remove veractiy first
MU3D_Video_Level_Data <- MU3D_Video_Level_Data0[,-3]
str(MU3D_Video_Level_Data)

pie(table(MU3D_Target_Level_Data0$Age),radius = 1, cex = 0.4)
library(dplyr)
library(forcats)
library(ggplot2)
#check race variable 
dataset <- data.frame(table(MU3D_Target_Level_Data0$Age))
colnames(dataset) <- c("Age", "Count")
ggplot(dataset, aes(x = "", y = Count, fill = Age)) +
  geom_col(width = 1) + 
  coord_polar(theta = "y") +
  xlab(NULL)

p.cor = cor(MU3D_Video_Level_Data[,-2])
corrplot.mixed(p.cor, title(main = "Correlation plot for features"),mar=c(0,0,1,0))
