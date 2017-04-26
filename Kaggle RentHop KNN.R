library(tibble)
library(stringr)
library(data.table)
library(quanteda)
library(readr)
library(class)
library(nnet)
library(caret)


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




train$interest_level <- as.factor(train$interest_level)

y_train <- train$interest_level

yIND <- class.ind(train$interest_level)

listing_id <- test$listing_id

train <- cbind(train, yIND)
test$low <- test$medium <- test$high <- -1

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



# find names of variables in set for formula 
modelFeats <- setdiff(feats, features_to_drop)
# remove low, medium, and high from right hand side of formula
modelFeats <- modelFeats[-c(10, 11, 12)]
# convert modelFeats to data frame so names can be inspected
modelFeatsDF <- data.frame(modelFeats)
# write the formula
a <- as.formula(paste("low + medium + high ~ ", paste(sprintf("`%s`", modelFeats), collapse = " + ")))




old2 <- Sys.time()

knnModel <- knn3Train(train, test, cl = y_train, k = 5, prob = TRUE)

new2 <- Sys.time() - old2
print(new2)

knnModelDF <- data.frame(knnModel[attr("prob")])



old3 <- Sys.time()

knnModel2 <- knn3(x = train, y = y_train, k = 5)

new3 <- Sys.time() - old3
print(new3)

old4 <- Sys.time()

knnPredict <- predict(knnModel2, test, type = "prob")

new4 <- Sys.time() - old4
print(new4)

knnPrediction <- data.frame(cbind(listing_id, knnPredict))

write.csv(knnPrediction,"knnPrediction.csv", row.names = FALSE)
