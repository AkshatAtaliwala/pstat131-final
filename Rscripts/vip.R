library(tidymodels) #tidymodels /tidyverse for model building and standardized code workflow |
library(tidyverse)
library(janitor) # for clean names
library(ggplot2) # for EDA / General plotting
library(corrplot) # for correlation plot
library(glmnet) # ridge / lasso regression
library(vip) # visualize variable importance
library(randomForest) # Random Forest
library(xgboost) # boosted trees
library(kernlab) # support vector machines
library(skimr) # EDA 
tidymodels_prefer()

fifa <- read_csv("data/fifa_clean.csv")

set.seed(3478)
data_split <- initial_split(fifa, 
                            prop = 0.7,
                            strata = overall_rating)

train <- training(data_split)
test <- testing(data_split)

folds <- vfold_cv(data = train, 
                  v = 5, 
                  strata = overall_rating)

fifa_recipe <- recipe(overall_rating ~ ., 
                      data = train) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())


random_forest_model <- rand_forest(mtry=tune(), trees=tune(), min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("randomForest", importance = TRUE)

random_forest_wflow <- workflow() %>% 
  add_recipe(fifa_recipe) %>% 
  add_model(random_forest_model)

random_forest_tune_res <- read_rds("results/random_forest.rds")

random_forest_final <- finalize_workflow(random_forest_wflow, best_random_forest)
random_forest_final_fit <- fit(random_forest_final, data = train)

write_rds(random_forest_final_fit, "results/vip.rds")