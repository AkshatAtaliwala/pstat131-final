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

elastic_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

elastic_wflow <- workflow() %>% 
  add_recipe(fifa_recipe) %>% 
  add_model(elastic_model)

elastic_penalty_grid <- grid_regular(penalty(range = c(-7, 7)),
                                     mixture(range = c(0,1)),
                                     levels = 12)

elastic_tune_res <- tune_grid(elastic_wflow,
                              resamples = folds,
                              grid = elastic_penalty_grid)

write_rds(elastic_tune_res, "results/elastic_net.rds")

