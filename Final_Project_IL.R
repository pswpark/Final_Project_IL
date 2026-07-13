## Data analysis preparation

### library loading 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

#Main library loading 
library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
library(dslabs)
library(lubridate)

### Data download from Git/Kaggle 
url <- "https://raw.githubusercontent.com/pswpark/Final_Project_IL/main/indian_liver_patient.csv"
liver_data <- read.csv(url)

### Data preparation

# Factorisation and creation of disease column instead of Dataset
liver_data <- liver_data %>% 
  mutate(Disease = as.factor(Dataset),
         Gender = as.factor(Gender)) %>% 
  select(-Dataset)

# R variable name error will occur if we use 1 and 2 as factor names when we create ensemble. 
liver_data$Disease <- factor(
  liver_data$Disease,
  levels = c(1, 2),
  labels = c("Yes", "No")
)


### Creation of final holdout set and separate train/test sets afterwards

# Final hold-out test set will be 15% of the total data based on Disease column 
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = liver_data$Disease, times = 1, p = 0.15, list = FALSE)
liv <- liver_data[-test_index,]
final_holdout_test <- liver_data[test_index,]

# Splitting of train/test sets with 15% in the test set 

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = liv$Disease, times = 1, p = 0.15, list = FALSE) #note the data is from liv, not liver_data
train_set <- liv[-test_index,]
test_set <- liv[test_index,]

### Further data exploration using visualisation 

#### Age 

train_set %>% 
  ggplot(aes(Age, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  labs(
    x = "Age",
    y = "Density",
    title = "Age Distribution by Disease Status (Boxplot)"
  )

#### Gender 

#Gender and its relationship to Disease

train_set %>%
  count(Gender, Disease) %>%
  group_by(Gender) %>%
  mutate(prop = n / sum(n))

#### Bilirubins 

train_set %>% 
  ggplot(aes(Total_Bilirubin, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  scale_x_log10() + 
  labs(
    x = "Total_Bilirubin",
    y = "Density",
    title = "Total_Bilirubin Distribution by Disease Status (Boxplot, log)"
  )

train_set %>% 
  ggplot(aes(Direct_Bilirubin, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  scale_x_log10() + 
  labs(
    x = "Direct_Bilirubin",
    y = "Density",
    title = "Direct_Bilirubin Distribution by Disease Status (Boxplot,log)"
  )

#### Other factors 

#ALP - cholestasis
train_set %>% 
  ggplot(aes(Alkaline_Phosphotase, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  scale_x_log10() + 
  labs(
    x = "Alkaline_Phosphotase",
    y = "Density",
    title = "Alkaline_Phosphotase Distribution by Disease Status (Boxplot, Log)"
  )

#ALT - hepatocellular 
train_set %>% 
  ggplot(aes(Alamine_Aminotransferase, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  scale_x_log10() + 
  labs(
    x = "Alamine_Aminotransferase",
    y = "Density",
    title = "Alamine_Aminotransferase Distribution by Disease Status (Boxplot, Log)"
  )
#AST- hepatocellular
train_set %>% 
  ggplot(aes(Aspartate_Aminotransferase, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  scale_x_log10() + 
  labs(
    x = "Aspartate_Aminotransferase",
    y = "Density",
    title = "Aspartate_Aminotransferase Distribution by Disease Status (Boxplot, Log)"
  )

#Total protein 
train_set %>% 
  ggplot(aes(Total_Protiens, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  labs(
    x = "Total_Protiens",
    y = "Density",
    title = "Total_Protiens Distribution by Disease Status (Boxplot)"
  )

# Albumin 
train_set %>% 
  ggplot(aes(Albumin, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  labs(
    x = "Total_Protiens",
    y = "Density",
    title = "Total_Protiens Distribution by Disease Status (Boxplot)"
  )

# Albumin_and_Globulin_Ratio
train_set %>% 
  ggplot(aes(Albumin_and_Globulin_Ratio, fill=Disease)) + geom_boxplot(alpha=0.4) + 
  labs(
    x = "Albumin_and_Globulin_Ratio",
    y = "Density",
    title = "Albumin_and_Globulin_Ratio Distribution by Disease Status (Boxplot)"
  )

## Handling of NA 
train_set <- na.omit(train_set)  
test_set <- na.omit(test_set)


## Model 1: logistic regression (glm)

# train on train_set
train_glm <- train(Disease ~ ., method = "glm", data = train_set)

#predict on test set based on train set
y_hat_glm <- predict(train_glm, test_set, type = "raw")

#metric calculation 
cm_glm <- confusionMatrix(y_hat_glm, test_set$Disease)


## Model 2: KNN

train_knn <- train(Disease ~ ., method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(1, 51, 2)))

cm_knn <- confusionMatrix(data = predict(train_knn, test_set, type = "raw"), 
                          reference = test_set$Disease)


## Model 3: gamLoess 

if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
modelLookup("gamLoess")

#trying to find best span for the model with tuneGrid

#train control = 10 fold validations - same cross-validation folds
train_control <- trainControl(
  method = "cv",
  number = 10
)

#this will take about 30 seconds
train_gam <- train(Disease ~ ., data = train_set, method = "gamLoess",
                   trControl = train_control,
                   tuneGrid = data.frame(span = seq(0.3, 1.0, by = 0.1), degree =1)
)

#predict on test set based on train set
y_hat_gam <- predict(train_gam, test_set)

#metric calculation 
cm_gam <- confusionMatrix(y_hat_gam, test_set$Disease)


## Model 4: Random Forest 

if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

#mtry values of c(2, 4, 6, 8) given about 10 features being assessed. 

train_rf <- train(Disease ~ ., data = train_set,  method = "rf",
                  trControl = train_control, #train control = 10 fold validations previously defined 
                  tuneGrid = data.frame(
                    mtry = c(2, 4, 6, 8)
                  )
)

y_hat_rf <- predict(train_rf, test_set)

#metric calculation 
cm_rf <- confusionMatrix(y_hat_rf, test_set$Disease)

## Summary table of individual models 

results <- data.frame(
  Model = c("Logistic Regression", "KNN", "gamLoess", "Random Forest"),
  Precision = c(
    cm_glm$byClass["Pos Pred Value"],
    cm_knn$byClass["Pos Pred Value"],
    cm_gam$byClass["Pos Pred Value"],
    cm_rf$byClass["Pos Pred Value"]
  ),
  Recall = c(
    cm_glm$byClass["Sensitivity"],
    cm_knn$byClass["Sensitivity"],
    cm_gam$byClass["Sensitivity"],
    cm_rf$byClass["Sensitivity"]
  ),
  Specificity = c(
    cm_glm$byClass["Specificity"],
    cm_knn$byClass["Specificity"],
    cm_gam$byClass["Specificity"],
    cm_rf$byClass["Specificity"]
  ),
  F1 = c(
    cm_glm$byClass["F1"],
    cm_knn$byClass["F1"],
    cm_gam$byClass["F1"],
    cm_rf$byClass["F1"]
  ),
  Balanced_Accuracy = c(
    cm_glm$byClass["Balanced Accuracy"],
    cm_knn$byClass["Balanced Accuracy"],
    cm_gam$byClass["Balanced Accuracy"],
    cm_rf$byClass["Balanced Accuracy"]
  ),
  Accuracy = c(
    cm_glm$overall["Accuracy"],
    cm_knn$overall["Accuracy"],
    cm_gam$overall["Accuracy"],
    cm_rf$overall["Accuracy"]
  )
)

results


## Ensemble with stacked models  

# Use caretEnsemble package 
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")

# New model list creation combining all models with specific commands: 

train_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final"
)

model_list <- caretList(
  Disease ~ .,
  data = train_set,
  trControl = train_control,
  tuneList = list(
    glm = caretModelSpec(method = "glm"),
    knn = caretModelSpec(
      method = "knn",
      tuneGrid = data.frame(k = seq(1, 51, 2)),
      preProcess = c("center", "scale")
    ),
    gam = caretModelSpec(
      method = "gamLoess",
      tuneGrid = expand.grid(
        span = seq(0.3, 1, 0.1),
        degree = 1
      )
    ),
    rf = caretModelSpec(
      method = "rf",
      tuneGrid = data.frame(mtry = c(2, 4, 6, 8))
    )
  )
)

# stacking using glm 
stack_model <- caretStack(
  model_list,
  method = "glm"
)

#y_hat prediction
y_hat_ensemble <- predict(stack_model,newdata = test_set)

#Transform this to a factored outcome 
y_hat_class <- ifelse(
  y_hat_ensemble$Yes >= 0.5,
  "Yes",
  "No"
)

y_hat_class <- factor(
  y_hat_class,
  levels = levels(test_set$Disease)
)

cm_ensemble <- confusionMatrix(
  y_hat_class,
  test_set$Disease,
  positive = "Yes"
)

cm_ensemble

results <- bind_rows(results,
                     tibble(Model = "Ensemble of 4 Models", 
                            Precision = cm_ensemble$byClass["Pos Pred Value"],
                            Recall = cm_ensemble$byClass["Sensitivity"],
                            Specificity = cm_ensemble$byClass["Specificity"],
                            F1 = cm_ensemble$byClass["F1"],
                            Balanced_Accuracy = cm_ensemble$byClass["Balanced Accuracy"],
                            Accuracy = cm_ensemble$overall["Accuracy"]))

results 


## Test on final hold out set 

# glm 
y_hat_glm_2 <- predict(train_glm, final_holdout_test, type = "raw")
cm_glm_2 <- confusionMatrix(y_hat_glm_2, final_holdout_test$Disease)

cm_glm_2

# knn
cm_knn_2 <- confusionMatrix(data = predict(train_knn, final_holdout_test, type = "raw"), 
                            reference = final_holdout_test$Disease)
# gamLoess
y_hat_gam_2 <- predict(train_gam, final_holdout_test)
cm_gam_2 <- confusionMatrix(y_hat_gam_2, final_holdout_test$Disease)

# rf 
y_hat_rf_2 <- predict(train_rf, final_holdout_test)
cm_rf_2 <- confusionMatrix(y_hat_rf_2, final_holdout_test$Disease)

# ensemble 
y_hat_ensemble_2 <- predict(stack_model, newdata = final_holdout_test)

#Transform this to a factored outcome 
y_hat_class_2 <- ifelse(
  y_hat_ensemble_2$Yes >= 0.5,
  "Yes",
  "No"
)

y_hat_class_2 <- factor(
  y_hat_class_2,
  levels = levels(final_holdout_test$Disease)
)

cm_ensemble_2 <- confusionMatrix(
  y_hat_class_2,
  final_holdout_test$Disease,
  positive = "Yes"
)

cm_ensemble_2

results_2 <- data.frame(
  Model = c("Logistic Regression", "KNN", "gamLoess", "Random Forest", "Ensemble of 4 Models"),
  Precision = c(
    cm_glm_2$byClass["Pos Pred Value"],
    cm_knn_2$byClass["Pos Pred Value"],
    cm_gam_2$byClass["Pos Pred Value"],
    cm_rf_2$byClass["Pos Pred Value"], 
    cm_ensemble_2$byClass["Pos Pred Value"]
  ),
  Recall = c(
    cm_glm_2$byClass["Sensitivity"],
    cm_knn_2$byClass["Sensitivity"],
    cm_gam_2$byClass["Sensitivity"],
    cm_rf_2$byClass["Sensitivity"],
    cm_ensemble_2$byClass["Sensitivity"]
  ),
  Specificity = c(
    cm_glm_2$byClass["Specificity"],
    cm_knn_2$byClass["Specificity"],
    cm_gam_2$byClass["Specificity"],
    cm_rf_2$byClass["Specificity"],
    cm_ensemble_2$byClass["Specificity"]
  ),
  F1 = c(
    cm_glm_2$byClass["F1"],
    cm_knn_2$byClass["F1"],
    cm_gam_2$byClass["F1"],
    cm_rf_2$byClass["F1"],
    cm_ensemble_2$byClass["F1"]
  ),
  Balanced_Accuracy = c(
    cm_glm_2$byClass["Balanced Accuracy"],
    cm_knn_2$byClass["Balanced Accuracy"],
    cm_gam_2$byClass["Balanced Accuracy"],
    cm_rf_2$byClass["Balanced Accuracy"],
    cm_ensemble_2$byClass["Balanced Accuracy"]
  ),
  Accuracy = c(
    cm_glm_2$overall["Accuracy"],
    cm_knn_2$overall["Accuracy"],
    cm_gam_2$overall["Accuracy"],
    cm_rf_2$overall["Accuracy"],
    cm_ensemble_2$overall["Accuracy"]
  )
)

print(results_2)
