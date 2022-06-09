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

tree_model <- decision_tree(cost_complexity = tune()) %>%
  set_mode("regression") %>%
  set_engine("rpart")

tree_wflow <- workflow() %>% 
  add_recipe(fifa_recipe) %>% 
  add_model(tree_model)

tree_cost_grid <- grid_regular(cost_complexity(range = c(-10, -1)), levels = 10)

tree_tune_res <- tune_grid(
  tree_wflow, 
  resamples = folds, 
  grid = tree_cost_grid)

write_rds(tree_tune_res, "results/decision_tree.rds")

