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

fifa <- read_csv("data/fifa.csv")

fifa <- fifa %>% clean_names()

keeps <- c("overall_rating", "club_team", "club_position", "age", "height_cm", "weight_kgs","wage_euro", "nationality",
           "preferred_foot", "international_reputation_1_5", "weak_foot_1_5", "skill_moves_1_5", "work_rate", "crossing", "finishing",
           "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "freekick_accuracy", "long_passing","ball_control",
           "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots",
           "aggression", "interceptions", "positioning", "vision", "penalties","composure", "marking", "standing_tackle",
           "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes") 
fifa <- fifa[, keeps]

fifa <- drop_na(fifa)

exclusions <- c('Andorra', 'Yemen', 'Nicaragua', 'South Sudan', 'Malta', 'Eritrea', 'Liberia',
                'St Lucia', 'United Arab Emirates', 'Guam', 'Indonesia', 'New Grenada','Fiji',
                'Barbados','Guatemala', 'Faroe Islands', 'Korea DPR','Kuwait','Tanzania',
                'Hong Kong','St Kitts Nevis','Tanzania', "Côte d'Ivoire", 'Oman', 'Chad',
                "São Tomé & Príncipe", "Rwanda", "Papua New Guinea", "Latvia", "Vietnam",
                'Azerbaijan', 'Ethiopia', 'Jordan', "New Caledonia", "Dominican Republic",
                "Liechtenstein", 'Montserrat', "Afghanistan", "Uzbekistan", "Palestine",
                "Suriname","Comoros", "Grenada", "Libya", "Thailand", "Philippines",
                "Antigua & Barbuda", "Central African Rep.", "Cuba", "Guyana")

fifa <- fifa[-which(fifa$nationality %in% exclusions), ] #17598 rows

fifa <- fifa[-which(fifa$club_team %in% exclusions), ] #17597 rows

fifa <- fifa[-which(fifa$club_team == fifa$nationality), ] #17483 rows

write_csv(fifa, "data/fifa_clean.csv")
