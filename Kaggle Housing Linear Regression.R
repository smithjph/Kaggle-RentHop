library(data.table)
library(Metrics)
library(Matrix)
library(mice)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)




#set working directory
setwd('/users/thesmithfamily/desktop/kaggle/ames')

#define submission file for later
SUBMISSION = "/users/thesmithfamily/desktop/kaggle/ames/sample_submission.csv"

#load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")


#Row binding train & test set for feature engineering
train_test = bind_rows(train, test)


#remove houses with more than 4000 square feet as recommended by the dataset creator, https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
#train_test <- train_test[which(train_test$GrLivArea < 4000),]
#train <- train[which(train$GrLivArea < 4000),]
#test <- test[which(test$GrLivArea < 4000),]


#set number of rows in training set
ntrain = nrow(train)


#set variable to be predicted
y_train <- train$SalePrice

#plot Sale Prices to get a feel for the data
hist(y_train, breaks = 100, xlim = c(30000, 800000), xlab = "SalePrice")
#taking the log of SalePrice appears to fix the skewness of the data
hist(log(y_train), breaks = 100, xlab = "log(SalePrice)")

#recode y_train to log(y_train)
y_train <- log(y_train)




#Remove Id since of no use
train$Id = NULL
train$SalePrice = NULL
test$Id = NULL
train_test$SalePrice = NULL






#graph distributions of continuous variables to check for normality
hist(train_test$MasVnrArea, breaks = 100)
hist(log(train_test$MasVnrArea), breaks = 100)

hist(train_test$BsmtFinSF1, breaks = 100)
hist(log(train_test$BsmtFinSF1), breaks = 100)

hist(train_test$BsmtFinSF2, breaks = 100)
hist(log(train_test$BsmtFinSF2), breaks = 100)

hist(train_test$BsmtUnfSF, breaks = 100)
hist(log(train_test$BsmtUnfSF), breaks = 100)

hist(train_test$TotalBsmtSF, breaks = 100)
hist(log(train_test$TotalBsmtSF), breaks = 100)

#hist(train_test$X1stFlrSF, breaks = 100) - looks good, no transformation needed

hist(train_test$X2ndFlrSF, breaks = 100)
hist(log(train_test$X2ndFlrSF), breaks = 100)

hist(train_test$GrLivArea, breaks = 100)
hist((train_test$GrLivArea)**(1/3), breaks = 100)

hist(train_test$GarageArea, breaks = 100)
hist(log(train_test$GarageArea), breaks = 100)

hist(train_test$WoodDeckSF, breaks = 100)
hist(log(train_test$WoodDeckSF), breaks = 100)

hist(train_test$OpenPorchSF, breaks = 100)
hist(log(train_test$OpenPorchSF), breaks = 100)

hist(train_test$EnclosedPorch, breaks = 100)
hist(log(train_test$EnclosedPorch), breaks = 100)

hist(train_test$ScreenPorch, breaks = 100)
hist(log(train_test$ScreenPorch), breaks = 100)

hist(train_test$LotFrontage, breaks = 100)
hist(log(train_test$LotFrontage), breaks = 100)

hist(train_test$LotArea, breaks = 100)
hist(log(train_test$LotArea), breaks = 100)





#create new variables
#create a total SF variable from other Square Footage variables
train_test$TotalSF = rowSums(cbind(train_test$TotalBsmtSF, train_test$X1stFlrSF, train_test$X2ndFlrSF))


attach(train_test)
#create a dummy variable for the three most expensive neighborhoods
train_test$NeighborhoodDummy <- ifelse(Neighborhood == "NoRidge", 1, ifelse(Neighborhood == "NridgHt", 1, ifelse(Neighborhood == "Somerst", 1, 0)))


#convert MSSubClass from integer to factor
train_test$MSSubClass[train_test$MSSubClass == "20"] <- "SC20"
train_test$MSSubClass[train_test$MSSubClass == "30"] <- "SC30"
train_test$MSSubClass[train_test$MSSubClass == "40"] <- "SC40"
train_test$MSSubClass[train_test$MSSubClass == "45"] <- "SC45"
train_test$MSSubClass[train_test$MSSubClass == "50"] <- "SC50"
train_test$MSSubClass[train_test$MSSubClass == "60"] <- "SC60"
train_test$MSSubClass[train_test$MSSubClass == "70"] <- "SC70"
train_test$MSSubClass[train_test$MSSubClass == "75"] <- "SC75"
train_test$MSSubClass[train_test$MSSubClass == "80"] <- "SC80"
train_test$MSSubClass[train_test$MSSubClass == "85"] <- "SC85"
train_test$MSSubClass[train_test$MSSubClass == "90"] <- "SC90"
train_test$MSSubClass[train_test$MSSubClass == "120"] <- "SC120"
train_test$MSSubClass[train_test$MSSubClass == "150"] <- "SC150"
train_test$MSSubClass[train_test$MSSubClass == "160"] <- "SC160"
train_test$MSSubClass[train_test$MSSubClass == "180"] <- "SC180"
train_test$MSSubClass[train_test$MSSubClass == "190"] <- "SC190"


#recode NA for Alley to "None"
AlleyLevels <- levels(train_test$Alley)
AlleyLevels[length(AlleyLevels) + 1] <- "None"
train_test$Alley <- factor(train_test$Alley, levels = AlleyLevels)
train_test$Alley[is.na(train_test$Alley)] <- "None"


#recode NA for Fence to "None"
FenceLevels <- levels(train_test$Fence)
FenceLevels[length(FenceLevels) + 1] <- "None"
train_test$Fence <- factor(train_test$Fence, levels = FenceLevels)
train_test$Fence[is.na(train_test$Fence)] <- "None"


#recode NA for BsmtCond to "None"
BsmtCondLevels <- levels(train_test$BsmtCond)
BsmtCondLevels[length(BsmtCondLevels) + 1] <- "None"
train_test$BsmtCond <- factor(train_test$BsmtCond, levels = BsmtCondLevels)
train_test$BsmtCond[is.na(train_test$BsmtCond)] <- "None"


#recode NA for BsmtExposure to "None"
BsmtExposureLevels <- levels(train_test$BsmtExposure)
BsmtExposureLevels[length(BsmtExposureLevels) + 1] <- "None"
train_test$BsmtExposure <- factor(train_test$BsmtExposure, levels = BsmtExposureLevels)
train_test$BsmtExposure[is.na(train_test$BsmtExposure)] <- "None"


#recode NA for BsmtFinType1 to "None"
BsmtFinType1Levels <- levels(train_test$BsmtFinType1)
BsmtFinType1Levels[length(BsmtFinType1Levels) + 1] <- "None"
train_test$BsmtFinType1 <- factor(train_test$BsmtFinType1, levels = BsmtFinType1Levels)
train_test$BsmtFinType1[is.na(train_test$BsmtFinType1)] <- "None"


#recode NA for BsmtFinType2 to "None"
BsmtFinType2Levels <- levels(train_test$BsmtFinType2)
BsmtFinType2Levels[length(BsmtFinType2Levels) + 1] <- "None"
train_test$BsmtFinType2 <- factor(train_test$BsmtFinType2, levels = BsmtFinType2Levels)
train_test$BsmtFinType2[is.na(train_test$BsmtFinType2)] <- "None"


#recode NA for FireplaceQu to "None"
FireplaceQuLevels <- levels(train_test$FireplaceQu)
FireplaceQuLevels[length(FireplaceQuLevels) + 1] <- "None"
train_test$FireplaceQu <- factor(train_test$FireplaceQu, levels = FireplaceQuLevels)
train_test$FireplaceQu[is.na(train_test$FireplaceQu)] <- "None"


#replace NA with 0 where it makes sense
train_test$MasVnrArea[is.na(train_test$MasVnrArea)] <- 0



#create transformation variables
train_test$MasVnrArea <- log(train_test$MasVnrArea)

train_test$BsmtFinSF1 <- log(train_test$BsmtFinSF1)

train_test$BsmtFinSF2 <- log(train_test$BsmtFinSF2)

train_test$BsmtUnfSF <- log(train_test$BsmtUnfSF)

train_test$TotalBsmtSF <- log(train_test$TotalBsmtSF)

train_test$X2ndFlrSF <- log(train_test$X2ndFlrSF)

train_test$GrLivArea <- log(train_test$GrLivArea)

train_test$GarageArea <- log(train_test$GarageArea)

train_test$WoodDeckSF <- log(train_test$WoodDeckSF)

train_test$OpenPorchSF <- log(train_test$OpenPorchSF)

train_test$EnclosedPorch <- log(train_test$EnclosedPorch)

train_test$ScreenPorch <- log(train_test$ScreenPorch)

train_test$LotFrontage <- log(train_test$LotFrontage)

train_test$LotArea <- log(train_test$LotArea)


#HeatingQC - dummy for ExAndGd/Not
train_test$HeatingQCDummy <- ifelse(HeatingQC == "Ex", 1, train_test$HeatingQCDummy <- ifelse(HeatingQC == "Gd", 1, 0))


#SaleCondition - dummy for Normal/Not
train_test$SaleConditionDummy <- ifelse(SaleCondition == "Normal", 1, 0)


#Condition1 - dummy for Norm/NotNorm
train_test$Condition1Dummy <- ifelse(Condition1 == "Norm", 1, 0)


#Foundation - dummy for PConc/NotPConc
train_test$FoundationDummy <- ifelse(Foundation == "PConc", 1, 0)


#ExterCond - dummy for ExAndGd/NotExAndGd
train_test$ExterCondDummy <- ifelse(ExterCond == "Ex", 1, train_test$ExterCondDummy <- ifelse(ExterCond == "Gd", 1, 0))


#LandContour - dummy variable for Lvl/notLvl
train_test$LandContourDummy <- ifelse(LandContour == "Lvl", 1, 0)


#YearRemodAdd - dummy variable for remodel within the past year of selling
train_test$YearRemodAddDummy <- ifelse(YearRemodAdd == YrSold, 1, 0)


#NewHouseDummy - dummy variable for whether a house was built in the year it sold
train_test$NewHouseDummy <- ifelse(YearBuilt == YrSold, 1, 0)


#HouseAge - variable representing the age of the house when it sold
train_test$HouseAge <- train_test$YrSold - train_test$YearBuilt


#GarageAge - variable representing the age of the garage when the house was sold
train_test$GarageAge <- train_test$YrSold - train_test$GarageYrBlt


#TimeRemod - variable representing number of years since last remodel when the house sold
train_test$TimeRemod <- train_test$YrSold - train_test$YearRemodAdd


#IsRemod - dummy variable representing whether there has been a remodel on the house
train_test$IsRemod <- ifelse(train_test$YearBuilt == train_test$YearRemodAdd, 0, 1)


#NumBath - variable representing the total number of bathrooms
train_test$NumBath <- (0.5 * train_test$HalfBath) + (0.5 * train_test$BsmtHalfBath) +
  train_test$FullBath + train_test$BsmtFullBath


#NumRooms - variable representing the total number of rooms + bathrooms
train_test$NumRooms <- train_test$TotRmsAbvGrd + train_test$FullBath + train_test$HalfBath



#polynomials of top continuous features according to gain on importance model
train_test$TotalSF2 <- train_test$TotalSF**2
train_test$TotalSF3 <- train_test$TotalSF**3
train_test$TotalSFsqrt <- sqrt(train_test$TotalSF)
train_test$X2ndFlrSF2 <- train_test$X2ndFlrSF**2
train_test$X2ndFlrSF3 <- train_test$X2ndFlrSF**3
#train_test$X2ndFlrSFsqrt <- sqrt(train_test$X2ndFlrSF)
train_test$GarageArea2 <- train_test$GarageArea**2
train_test$GarageArea3 <- train_test$GarageArea**3
#train_test$GarageAreasqrt <- sqrt(train_test$GarageArea)
train_test$X1stFlrSF2 <- train_test$X1stFlrSF**2
train_test$X1stFlrSF3 <- train_test$X1stFlrSF**3
train_test$X1stFlrSFsqrt <- sqrt(train_test$X1stFlrSF)
train_test$LotFrontage2 <- train_test$LotFrontage**2
train_test$LotFrontage3 <- train_test$LotFrontage**3
train_test$LotFrontagesqrt <- sqrt(train_test$LotFrontage)









#get names of all variables in full data set
features=names(train_test)

#convert character into integer
for(f in features){
  if(class(train_test[[f]]) == "character"){
    levels = sort(unique(train_test[[f]]))
    train_test[[f]] = as.integer(factor(train_test[[f]],levels = levels))
  }
}



#features to exclude, Utilities, Electrical, PoolQC, and MiscFeature excluded due to low variance
#SalePrice excluded to allow XGBoost to work properly
features_to_drop <- c("Utilities","Electrical","PoolQC","MiscFeature",
                      "SalePrice")


#splitting whole data back again minus the dropped features
train_x = train_test[1:ntrain,!(features) %in% features_to_drop]
test_x = train_test[(ntrain+1):nrow(train_test),!(features) %in% features_to_drop]


#convert into numeric for XGBoost implementation
train_x[] <- lapply(train_x, as.numeric)
test_x[] <- lapply(test_x, as.numeric)

#replaces -inf with 0
train_x <- do.call(data.frame,lapply(train_x, function(x) replace(x, is.infinite(x), 0)))
test_x <- do.call(data.frame,lapply(test_x, function(x) replace(x, is.infinite(x), 0)))



#missing values imputation with mice
set.seed(256)
to_impute <- as.data.frame(test_x)
impute <- to_impute[c("MSZoning","Exterior1st","Exterior2nd","BsmtFinSF1",
                      "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath",
                      "KitchenQual","Functional","GarageCars","GarageArea","SaleType","TotalSF",
                      "GarageFinish","BsmtQual","GarageCond","GarageQual","GarageYrBlt",
                      "GarageType","LotFrontage","NumBath","GarageAge","MasVnrType")]#,
#"TotalSF2","TotalSF3","TotalSFsqrt","X2ndFlrSFsqrt","GarageArea2",
#"GarageArea3","GarageAreasqrt","LotFrontage2","LotFrontage3",
#"LotFrontagesqrt")]

#specify package complete is in to avoid confusion with tidyr
imputed <- mice::complete(mice(impute,m=5))


to_impute$MSZoning=imputed$MSZoning
to_impute$Utilities=imputed$Utilities
to_impute$Exterior1st=imputed$Exterior1st
to_impute$Exterior2nd=imputed$Exterior2nd
to_impute$BsmtFinSF1=imputed$BsmtFinSF1
to_impute$BsmtFinSF2=imputed$BsmtFinSF2
to_impute$BsmtUnfSF=imputed$BsmtUnfSF
to_impute$TotalBsmtSF=imputed$TotalBsmtSF
to_impute$BsmtHalfBath=imputed$BsmtHalfBath
to_impute$BsmtFullBath=imputed$BsmtFullBath
to_impute$KitchenQual=imputed$KitchenQual
to_impute$Functional=imputed$Functional
to_impute$GarageCars=imputed$GarageCars
to_impute$GarageArea=imputed$GarageArea
to_impute$GarageArea2=imputed$GarageArea2
to_impute$GarageArea3=imputed$GarageArea3
to_impute$SaleType=imputed$SaleType
to_impute$TotalSF=imputed$TotalSF
to_impute$TotalSF2=imputed$TotalSF2
to_impute$TotalSF3=imputed$TotalSF3
to_impute$TotalSFsqrt=imputed$TotalSFsqrt
to_impute$GarageFinish=imputed$GarageFinish
to_impute$BsmtQual=imputed$BsmtQual
to_impute$GarageCond=imputed$GarageCond
to_impute$GarageQual=imputed$GarageQual
to_impute$GarageYrBlt=imputed$GarageYrBlt
to_impute$GarageType=imputed$GarageType
to_impute$GarageAge=imputed$GarageAge
to_impute$LotFrontage=imputed$LotFrontage
to_impute$LotFrontage2=imputed$LotFrontage2
to_impute$LotFrontage3=imputed$LotFrontage3
to_impute$LotFrontagesqrt=imputed$LotFrontagesqrt
to_impute$NumBath=imputed$NumBath
to_impute$MasVnrType=imputed$MasVnrType


test_x = as.data.table(to_impute)
#colnames(test_x)[colSums(is.na(test_x)) > 0]


to_impute <- as.data.frame(train_x)
impute <- to_impute[c("MSZoning","Exterior1st","Exterior2nd","BsmtFinSF1",
                      "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath",
                      "KitchenQual","Functional","GarageCars","GarageArea","SaleType","TotalSF",
                      "GarageFinish","BsmtQual","GarageCond","GarageQual","GarageYrBlt",
                      "GarageType","LotFrontage","NumBath","GarageAge","MasVnrType")]


#specify package complete is in to avoid confusion with tidyr
imputed <- mice::complete(mice(impute,m=5))


to_impute$MSZoning=imputed$MSZoning
to_impute$Utilities=imputed$Utilities
to_impute$Exterior1st=imputed$Exterior1st
to_impute$Exterior2nd=imputed$Exterior2nd
to_impute$BsmtFinSF1=imputed$BsmtFinSF1
to_impute$BsmtFinSF2=imputed$BsmtFinSF2
to_impute$BsmtUnfSF=imputed$BsmtUnfSF
to_impute$TotalBsmtSF=imputed$TotalBsmtSF
to_impute$BsmtHalfBath=imputed$BsmtHalfBath
to_impute$BsmtFullBath=imputed$BsmtFullBath
to_impute$KitchenQual=imputed$KitchenQual
to_impute$Functional=imputed$Functional
to_impute$GarageCars=imputed$GarageCars
to_impute$GarageArea=imputed$GarageArea
to_impute$GarageArea2=imputed$GarageArea2
to_impute$GarageArea3=imputed$GarageArea3
to_impute$SaleType=imputed$SaleType
to_impute$TotalSF=imputed$TotalSF
to_impute$TotalSF2=imputed$TotalSF2
to_impute$TotalSF3=imputed$TotalSF3
to_impute$TotalSFsqrt=imputed$TotalSFsqrt
to_impute$GarageFinish=imputed$GarageFinish
to_impute$BsmtQual=imputed$BsmtQual
to_impute$GarageCond=imputed$GarageCond
to_impute$GarageQual=imputed$GarageQual
to_impute$GarageYrBlt=imputed$GarageYrBlt
to_impute$GarageType=imputed$GarageType
to_impute$GarageAge=imputed$GarageAge
to_impute$LotFrontage=imputed$LotFrontage
to_impute$LotFrontage2=imputed$LotFrontage2
to_impute$LotFrontage3=imputed$LotFrontage3
to_impute$LotFrontagesqrt=imputed$LotFrontagesqrt
to_impute$NumBath=imputed$NumBath
to_impute$MasVnrType=imputed$MasVnrType


train_x = as.data.table(to_impute)





# fit model
linearModel <- lm(y_train ~ ., train_x)


# predict and write to CSV file
submission = fread(SUBMISSION, colClasses = c("integer","numeric"))
submission$SalePrice = predict(linearModel, test_x)
submission$SalePrice = exp(submission$SalePrice)
write.csv(submission,"LinearModel.csv", row.names = FALSE)







