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

boosted_model <- boost_tree(trees=tune(), tree_depth = tune(), learn_rate = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

boosted_wflow <- workflow() %>% 
  add_recipe(fifa_recipe) %>% 
  add_model(boosted_model)

boosted_tree_grid <- grid_regular(trees(range = c(10, 1000)),
                                  tree_depth(range = c(1, 11)),
                                  learn_rate(range = c(0.001, 0.2)),
                                  levels = 10)

boosted_tune_res <- tune_grid(
  boosted_wflow, 
  resamples = folds, 
  grid = boosted_tree_grid)

write_rds(boosted_tune_res, "results/boosted_trees.rds")

