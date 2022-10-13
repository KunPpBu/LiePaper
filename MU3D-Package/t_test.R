

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

#level veractiy
levels0 <- unique(c(MU3D_Video_Level_Data0$Veracity, MU3D_Video_Level_Data0$Veracity))
#add veracity back
MU3D_Video_Level_Data0$Veracity <- factor(MU3D_Video_Level_Data0$Veracity,levels = levels0)
str(MU3D_Video_Level_Data0)

#################################
# t-test for variables - video
#################################
tests_list <- lapply(names(MU3D_Video_Level_Data0[,-c(1,3,14)]), function(x) t.test(as.formula(paste0(x, "~ Veracity")), data = MU3D_Video_Level_Data0))
result <- do.call(rbind, lapply(tests_list, `[[`, "estimate"))
pval <- sapply(tests_list, `[[`, "p.value")
result <- cbind(result, p.value = pval)
head(result)
result1 <- data.frame(result)
test_col <- paste0(names(MU3D_Video_Level_Data0[,-c(1,3,14)]), "~ Veracity")
result1$test.relation <- test_col
result2 <- result1[,c(4,1,2,3)]
dim(result2)
result2
