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

lm_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

poly_recipe <- fifa_recipe %>%
  step_poly(wage_euro, degree = tune()) 

poly_wflow <- workflow() %>%
  add_recipe(poly_recipe) %>%
  add_model(lm_model)

poly_degree_grid <- grid_regular(degree(range = c(1, 5)),levels = 5)

poly_tune_res <- tune_grid(
  object = poly_wflow, 
  resamples = folds, 
  grid = poly_degree_grid)

write_rds(poly_tune_res, "results/polynomial_regression.rds")

