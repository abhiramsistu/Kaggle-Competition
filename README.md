# Kaggle-Competition
Worked on a group project in Kaggle where we had to predict and test the accuracy of a dataset. we worked on enhancing the Churn Prediction using XGBoost using R Studio.
This is our R code -:


#Load the Data set
train <- read.csv("train.csv")
test <- read.csv("test.csv")

#Load the Packages
library(dplyr)
library(readr)
library(caret)
library(randomForest)
library(pROC)
library(xgboost)

#Check for and remove Missing Variables
sum(is.na(train))
sum(is.na(test))
train <- train %>%
mutate(daysInactive = ifelse(is.na(daysInactive), mean(daysInactive, na.rm = TRUE), daysInactive))

# Example: Creating a new feature "engagement Rate" as clicks / visits
train <- train %>%
  mutate(engagementRate = clicks / visits)

test <- test %>%
  mutate(engagementRate = clicks / visits)

# Normalizing/Standardizing features (example with engagement Rate)
train$engagementRate <- scale(train$engagementRate)
test$engagementRate <- scale(test$engagementRate)

# Create dummy variables for categorical features if any (assuming time Of Day is categorical)
train <- dummyVars("~ .", data = train) %>% predict(train)
test <- dummyVars("~ .", data = test) %>% predict(test)
set.seed(123) # for reproducibility
class(train)
str(train)
train_df <- as.data.frame(train)
index <- createDataPartition(train[, "churn"], p=0.8, list=FALSE)
trainSet <- train[index, ]
validSet <- train[-index, ]

library(readr) # For read_csv
train <- read_csv("train.csv")

# Assuming time Of Day is a categorical variable you want to make dummies for
dummies_train <- dummyVars("~ .", data = train)
train_transformed <- data.frame(predict(dummies_train, newdata = train))

dummies_test <- dummyVars("~ .", data = test)
test_transformed <- data.frame(predict(dummies_test, newdata = test))

# Verify that train_transformed and test_transformed are indeed data frames
str(train_transformed)
str(test_transformed)
set.seed(123) # For reproducibility
index <- createDataPartition(train_transformed$churn, p = 0.8, list = FALSE)
trainSet <- train_transformed[index, ]
validSet <- train_transformed[-index, ]

# Train a model, example with Random Forest
model <- randomForest(churn ~ ., data = trainSet, ntree = 100)

# Convert 'churn' to a factor if it's not already
train_transformed$churn <- as.factor(train_transformed$churn)

# Split the data again after making sure 'churn' is a factor
index <- createDataPartition(train_transformed$churn, p = 0.8, list = FALSE)
trainSet <- train_transformed[index, ]
validSet <- train_transformed[-index, ]

# Train the Random Forest model for classification
model <- randomForest(churn ~ ., data = trainSet, ntree = 100)

# Predict on validation set using the classification model
predictions <- predict(model, validSet, type = "prob")[,2]
aucResult <- roc(validSet$churn, predictions)
print(aucResult$auc)

# Retrain model on full training data
finalModel <- randomForest(churn ~ ., data = train, ntree = 100)

# Convert 'churn' to a factor if it's not already
train$churn <- as.factor(train$churn)

# Confirm that 'churn' is now a factor
str(train$churn)

# Retrain the Random Forest model for classification on the entire data set
finalModel <- randomForest(churn ~ ., data = train, ntree = 100)

# Confirm the model type
print(finalModel)

# Predict on the test set using the classification model
# Ensure that your test set has been pre processed in the same way as your training set
finalPredictions <- predict(finalModel, test_transformed, type = "prob")[,2]
test_transformed <- as.data.frame(test_transformed)

# Assuming the first column of test_transformed is 'id'
submission <- data.frame(id = test_transformed[,1], churn = finalPredictions)

# Write the submission data frame to a CSV file
write.csv(submission, "", row.names = FALSE)




