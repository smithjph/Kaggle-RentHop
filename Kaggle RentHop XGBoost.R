library(tibble)
library(stringr)
library(data.table)
library(quanteda)
library(xgboost)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(Matrix)


setwd("/users/thesmithfamily/desktop/kaggle/renthop")

# Correctly read in data, thanks to user Dan J on Kaggle,
# source: https://www.kaggle.com/danjordan/two-sigma-connect-rental-listing-inquiries/how-to-correctly-load-data-into-r/run/837274

# Load packages and data
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

train1 <- fromJSON("/users/thesmithfamily/desktop/kaggle/renthop/train.json")
train <- train1
test1 <- fromJSON("/users/thesmithfamily/desktop/kaggle/renthop/test.json")
test <- test1


# unlist every variable except 'photos' and 'features' and convert to tibble
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)

vars1 <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test$interest_level <- -1



# convert classes to integers for xgboost
class <- data.table(interest_level=c("low", "medium", "high"), class=c(0,1,2))
train <- merge(train, class, by="interest_level", all.x=TRUE, sort=F)
#train_test <- merge(train_test, class, by="interest_level", all.x=TRUE, sort=F)

y_train <- train$class

test$class <- -1

listing_id_test <- test$listing_id

# combine train and test sets
train_test <- rbind(train, test)

ntrain <- nrow(train)

# remove class variable to avoid leakage
train_test$class <- NULL



# convert to numeric
train_test$building_id <- as.integer(as.factor(train_test$building_id))

train_test$manager_id <- as.integer(as.factor(train_test$manager_id))

train_test$street_address <- as.integer(as.factor(train_test$street_address))

train_test$display_address <- as.integer(as.factor(train_test$display_address))



# create new variables
train_test$desc_wordcount <- str_count(train_test$description)

train_test$PricePerBed <- ifelse(!is.finite(train_test$price / train_test$bedrooms), -1, train_test$price/train_test$bedrooms)

train_test$PricePerBath <- ifelse(!is.finite(train_test$price / train_test$bathrooms), -1, train_test$price/train_test$bathrooms)

train_test$PricePerRoom <- ifelse(!is.finite(train_test$price / (train_test$bedrooms + train_test$bathrooms)), -1, train_test$price/(train_test$bedrooms+train_test$bathrooms))

train_test$BedPerBath <- ifelse(!is.finite(train_test$bedrooms / train_test$bathrooms), -1, train_test$bedrooms/train_test$bathrooms)

train_test$BedBathDiff <- train_test$bedrooms - train_test$bathrooms

train_test$BedBathSum <- train_test$bedrooms + train_test$bathrooms



# Photo counts
train_photos <- data.table(listing_id=rep(unlist(train$listing_id), lapply(train$photos, length)), features=unlist(train$photos))
test_photos <- data.table(listing_id=rep(unlist(test$listing_id), lapply(test$photos, length)), features=unlist(test$photos))
train_test_photos <- rbind(train_photos, test_photos)
rm(train_photos, test_photos);gc()

train_test_photos_summ <- train_test_photos[,.(photo_count=.N), by=listing_id]
train_test$photo_count <- train_test_photos_summ$photo_count[match(train_test$listing_id, train_test_photos_summ$listing_id)]
rm(train_test_photos, train_test_photos_summ);gc()



# unlist features
train_feats <- data.table(listing_id=rep(unlist(train$listing_id), lapply(train$features, length)), features=unlist(train$features))
test_feats <- data.table(listing_id=rep(unlist(test$listing_id), lapply(test$features, length)), features=unlist(test$features))
train_test_feats <- rbind(train_feats, test_feats)
rm(train_feats, test_feats);gc()
# merge Feature column
train_test_feats[,features:=gsub(" ", "_", paste0("feature_",trimws(char_tolower(features))))]
feats_summ <- train_test_feats[,.N, by=features]
train_test_feats_cast <- dcast.data.table(train_test_feats[!features %in% feats_summ[N<10, features]], listing_id ~ features, fun.aggregate = function(x) as.integer(length(x) > 0), value.var = "features")
train_test1 <- merge(train_test, train_test_feats_cast, by="listing_id", all.x=TRUE, sort=FALSE)
train_test_order <- as.numeric(train_test$listing_id)
train_test2 <- train_test1[match(train_test_order, train_test1$listing_id),]
train_test <- train_test2
rm(train_test_feats_cast);gc()


# fill missing values with -1
for (col in 1:ncol(train_test)){
  set(train_test, i=which(is.na(train_test[[col]])), j=col, value=-1)
}






#get names of all variables in full data set
feats <- names(train_test)


features_to_drop <- c("interest_level", "created", "description", "display_address",
                      "features", "photos")


# split whole data back to train and test
train <- train_test[1:ntrain,!(feats) %in% features_to_drop]
test <- train_test[(ntrain+1):nrow(train_test),!(feats) %in% features_to_drop]




# convert into numeric for XGBoost implementation
train[] <- lapply(train, as.numeric)
test[] <- lapply(test, as.numeric)



# convert train and test to sparse matrices
train <- Matrix(as.matrix(train, !(feats) %in% features_to_drop, with = FALSE), sparse = TRUE)
test <- Matrix(as.matrix(test, !(feats) %in% features_to_drop, with = FALSE), sparse = TRUE)


# split train into train_a and train_b for 2-fold stacking
inA <- createDataPartition(y_train, times = 1, p = 0.5, list= FALSE)
train_a <- data.frame(as.matrix(train))[inA,]
train_b <- data.frame(as.matrix(train))[-inA,]

y_train_a <- data.frame(y_train)[inA,]
y_train_b <- data.frame(y_train)[-inA,]

test_a <- data.frame(as.matrix(test))[inA,]
test_b <- data.frame(as.matrix(test))[-inA,]

# convert to xgb format
dtrain <- xgb.DMatrix(data = as.matrix(train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(test))

dtrain_a <- xgb.DMatrix(data = as.matrix(train_a), label = y_train_a)
dtrain_b <- xgb.DMatrix(data = as.matrix(train_b), label = y_train_b)

dtest_a <- xgb.DMatrix(data = as.matrix(test_a))
dtest_b <- xgb.DMatrix(data = as.matrix(test_b))

# set up parameters for xgboost
param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=3,
              eta = .02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .5
)




old <- Sys.time()

xgb1 <- xgboost(data = dtrain,
                params = param,
                #nrounds = 2710,              # max number of trees to build
                nrounds = 5000,
                #nrounds = 10,
                verbose = TRUE,              # will print performance information                           
                print_every_n = 1,           # will print all messages
                early_stopping_rounds = 10
                )

xgb_cv_1 = xgb.cv(params = param,
                  data = dtrain,
                  nrounds = 5000,
                  #nrounds = 10,
                  nfold = 5,                 # randomly partition original dataset into 5 equal size subsamples
                  prediction = TRUE,         # return the prediction using the final model 
                  showsd = TRUE,             # standard deviation of loss across folds
                  stratified = TRUE,         # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print_every_n = 1, 
                  early_stopping_rounds = 10
)



# write to csv file
sPreds_full <- as.data.table(t(matrix(predict(xgb1, dtest), nrow=3, ncol=nrow(dtest))))
colnames(sPreds_full) <- class$interest_level
fwrite(data.table(listing_id = listing_id_test, sPreds_full[,list(high,medium,low)]), "submissionXGB_full.csv")



#find importance of variables
model <- xgb.dump(xgb1, with_stats = T)

#get the feature names
names <- dimnames(data.matrix(train[,-1]))[[2]]

#compute the feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb1)
importance_names_train_test <- importance_matrix$Feature

#graph the importance
xgb.plot.importance(importance_matrix[1:nrow(importance_matrix),], main = "XGB_full")






# plot the rmse for the training and testing samples
xgb_cv_1$evaluation_log %>%
  select(-dplyr::contains("std")) %>%
  select(-dplyr::contains("iter")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, mlogloss, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = mlogloss, group = TestOrTrain, color = TestOrTrain))+#,
         #ylim = c(0,12)) + 
  geom_line() + 
  theme_bw()





xgb_a <- xgboost(data = dtrain_a,
                params = param,
                #nrounds = 2710,              # max number of trees to build
                nrounds = 5000,
                #nrounds = 10,
                verbose = TRUE,              # will print performance information                           
                print_every_n = 1,           # will print all messages
                early_stopping_rounds = 10
)

xgb_cv_a = xgb.cv(params = param,
                  data = dtrain_a,
                  nrounds = 5000,
                  #nrounds = 10,
                  nfold = 5,                 # randomly partition original dataset into 5 equal size subsamples
                  prediction = TRUE,         # return the prediction using the final model 
                  showsd = TRUE,             # standard deviation of loss across folds
                  stratified = TRUE,         # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print_every_n = 1, 
                  early_stopping_rounds = 10
)



# write to csv file
sPreds_a <- as.data.table(t(matrix(predict(xgb_a, dtest_a), nrow=3, ncol=nrow(dtest_a))))
colnames(sPreds_a) <- class$interest_level
fwrite(data.table(listing_id = listing_id_test[inA], sPreds_a[,list(high,medium,low)]), "submissionXGB_a.csv")





xgb_b <- xgboost(data = dtrain_b,
                params = param,
                #nrounds = 2710,              # max number of trees to build
                nrounds = 5000,
                #nrounds = 10,
                verbose = TRUE,              # will print performance information                           
                print_every_n = 1,           # will print all messages
                early_stopping_rounds = 10
)

xgb_cv_b = xgb.cv(params = param,
                  data = dtrain_b,
                  nrounds = 5000,
                  #nrounds = 10,
                  nfold = 5,                 # randomly partition original dataset into 5 equal size subsamples
                  prediction = TRUE,         # return the prediction using the final model 
                  showsd = TRUE,             # standard deviation of loss across folds
                  stratified = TRUE,         # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print_every_n = 1, 
                  early_stopping_rounds = 10
)



# write to csv file
sPreds_b <- as.data.table(t(matrix(predict(xgb_b, dtest_b), nrow=3, ncol=nrow(dtest_b))))
colnames(sPreds_b) <- class$interest_level
fwrite(data.table(listing_id = listing_id_test[-inA], sPreds_b[,list(high,medium,low)]), "submissionXGB_b.csv")



#find importance of variables
model_a <- xgb.dump(xgb_a, with_stats = T)

#get the feature names
names <- dimnames(data.matrix(train[,-1]))[[2]]

#compute the feature importance matrix
importance_matrix_a <- xgb.importance(names, model = xgb_a)
importance_names_train_test_a <- importance_matrix_a$Feature

#graph the importance
xgb.plot.importance(importance_matrix_a[1:nrow(importance_matrix_a),], main = "XGB_a")



#find importance of variables
model_b <- xgb.dump(xgb_b, with_stats = T)

#get the feature names
names <- dimnames(data.matrix(train[,-1]))[[2]]

#compute the feature importance matrix
importance_matrix_b <- xgb.importance(names, model = xgb_b)
importance_names_train_test_b <- importance_matrix_b$Feature

#graph the importance
xgb.plot.importance(importance_matrix_b[1:nrow(importance_matrix_b),], main = "XGB_b")


new <- Sys.time() - old
print(new)




# model averaging
xgbModel1 <- read.csv("submissionXGB_full.csv")
xgbModela <- read.csv("submissionXGB_a.csv")
xgbModelb <- read.csv("submissionXGB_b.csv")

temp <- cbind(xgbModel1, xgbModela, xgbModelb)
xgbAvg <- sapply(unique(colnames(temp)), function(x) rowMeans(temp[, colnames(temp) == x, drop=FALSE]))
xgbAvg <- data.frame(xgbAvg)

write.csv(xgbAvgDF, "XGBAvgSubmission.csv", row.names = FALSE)
nnAvgSubmission <- read.csv("nnetAvgSubmission.csv")



