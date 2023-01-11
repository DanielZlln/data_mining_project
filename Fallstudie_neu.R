# Pakete installieren
# install.packages("DataExplorer")
# install.packages("psych")
# install.packages("reticulate")
# devtools::install_github("tidymodels/tune")
# install.packages("doMC")
# install.packages("ranger")
# install.packages("xgboost")

# library laden
library(dplyr)
library(DataExplorer)
library(psych)
library(modeldata)
library(rpart)
library(rpart.plot)
library(probably)
library(tidyverse)
library(reticulate)
library(recipes)
library(finetune)
library(themis)
library(doMC)
library(ranger)
library(tidymodels)
tidymodels_prefer()

### Daten einlesen
load("heart_disease.RData")

str(heart_disease)
summary(heart_disease)

# Kaum NA -> löschen der Zeilen

# Hoher_Blutdruck max. 630 -> unlogisch muss 1 / 0
# Hoher_Cholspiegel max. 867 -> unlogisch muss 1 / 0
# BMI max. 98 sinnvoll? Bei 180cm über 300kg
# Rauchen max. 833 -> unlogisch muss 1 / 0
# Phys_Aktivität Wert 891 -> unlogisch muss 1 / 0
# ALLE ZEILEN LOESCHEN MIT DEN OBEREN WERTEN DA SO WENIG

introduce(heart_disease)
plot_intro(heart_disease)

heart_disease %>% 
  filter(Hoher_Cholspiegel > 1 | Hoher_Blutdruck > 1 |
           Rauchen > 1 | Phys_Aktivität > 1) %>% 
  select(Hoher_Blutdruck, Hoher_Cholspiegel, Rauchen, Phys_Aktivität) %>% 
  summarise(anzahl = n())
# Insgeasmt 16 Zeilen mit falschen Werten


# Herzinfarkt
heart_disease %>% 
  group_by(Herzinfarkt) %>% 
  summarise(anzahl = n())
# 2 Zeilen "unbekannt" löschen


# Infarkt
mean(heart_disease$Herzinfarkt == "Infarkt") 
# 9.5% der Patienten haben ein Infarkt -> Balancing

# Geschlecht
heart_disease %>% 
  group_by(Geschlecht) %>% 
  summarise(anzahl = n())
# Mehr Frauen als Männer
# Geschlecht = "unbekannt" löschen da nur 3 Einträge

# Alter
heart_disease %>% 
  group_by(Alter) %>% 
  summarise(anzahl = n())

# Einkommen
mode(heart_disease$Einkommen) # Einkommen ist "character"

heart_disease <- heart_disease %>% 
  mutate(Einkommen = str_remove(string = Einkommen, pattern = '\\<'),
         Einkommen = str_remove(string = Einkommen, pattern = '\\>'),
         Einkommen = str_remove(string = Einkommen, pattern = '\\ '))

str(heart_disease)

heart_disease %>% 
  group_by(Einkommen) %>% 
  summarise(anzahl = n()) %>% 
  arrange(desc(anzahl))

# Schulabschluss
heart_disease %>% 
  group_by(Bildung) %>% 
  summarise(anzahl = n()) %>% 
  arrange(desc(anzahl))

# Bildung = unbekannt, löschen da nur 4 Einträge

# Zeilen löschen die falsche Werte enthalten
heart_disease <- heart_disease %>% 
  filter(Alter != "unbekannt")

heart_disease <- heart_disease %>% 
  filter(Geschlecht != "unbekannt")

heart_disease <- heart_disease %>% 
  filter(Hoher_Blutdruck <= 1)

heart_disease <- heart_disease %>% 
  filter(Hoher_Cholspiegel <= 1)

heart_disease <- heart_disease %>% 
  filter(Rauchen <= 1)

heart_disease <- heart_disease %>% 
  filter(Phys_Aktivität <= 1)

heart_disease <- heart_disease %>% 
  filter(Herzinfarkt != "unbekannt")

heart_disease <- heart_disease %>% 
  filter(Bildung != "unbekannt")

heart_disease <- heart_disease %>% 
  filter(complete.cases(.))

# Spalte BMI 
boxplot(heart_disease$BMI~heart_disease$Alter) # Wieso so viele Ausreiser nach oben?
mean(heart_disease$BMI>30, na.rm = T) 

h = hist(heart_disease$BMI, breaks = 100)
ccat = cut(h$breaks, c(-Inf, 30, Inf))
plot(h, col=c("white","red")[ccat])

summary(heart_disease)
str(heart_disease)

# chr to factor
heart_disease <- mutate_if(heart_disease, is.character, as.factor)

table(Gesundheit = heart_disease$Allg_Gesundheit,
      Aktivität = cut(heart_disease$Phys_Gesundheit + 
                        heart_disease$Ment_Gesundheit,
                      breaks = c(0,10,20,30,40,50,60,Inf),
                      dig.lab = 5,
                      include.lowest = T))
# Gesundheitsstatus gut -> mehr kleine Werte

# ------------------------------------------------------------------------

# Create recipe
# heart_recipe_down <- 
#   recipe(Herzinfarkt ~., data = heart_disease) %>% 
#   step_tomek(Herzinfarkt, over_ratio = 1)

heart_recipe_up <-
  recipe(Herzinfarkt ~., 
         data = heart_disease) %>%
  step_smotenc(Herzinfarkt, 
                over_ratio = 1) %>% 
  step_normalize(all_nominal_predictors())

# Create second recipe with dummy 
# heart_recipe_down_num <- heart_recipe_down %>% 
#   step_dummy(all_nominal_predictors(), one_hot = TRUE)

heart_recipe_up_num <- heart_recipe_up %>% 
  step_dummy(all_nominal_predictors(), 
             one_hot = TRUE)

recipes <- list(lr_recipe = heart_recipe_up,
                dt_recipe = heart_recipe_up,
                rf_recipe = heart_recipe_up,
                xgb_recipe = heart_recipe_up_num)

# Datensatz aufteilen zwischen test und train
set.seed(123)
heart_split <- initial_split(heart_disease,
                             prop = 0.80,
                             strata = Herzinfarkt)

heart_split

dim(heart_split)

heart_train <- training(heart_split)
dim(heart_train)

heart_train %>% 
  count(Herzinfarkt) %>% 
  mutate(prop = n/sum(n))

heart_test <- testing(heart_split)
dim(heart_test)

heart_test %>% 
  count(Herzinfarkt) %>% 
  mutate(prop = n/sum(n))

# Resampling
# ------------------------------------------------------------------------------

# Cross Validation
set.seed(123)

heart_folds <- vfold_cv(heart_train)
heart_folds



# Modeling
# ------------------------------------------------------------------------------


# Logistic Regression
lr_model <- 
  logistic_reg(mode = "classification", 
               engine = "glm")

# Decision Tree
dt_model <- 
  decision_tree(mode = "classification",
                engine = "rpart",
                cost_complexity = tune(),
                min_n = tune())

# Random Forest
rf_model <- 
  rand_forest(mode = "classification",
              engine = "ranger",
              trees = tune())

# XGBoost
xgb_model <- 
  boost_tree(mode = "classification",
             engine = "xgboost",
             learn_rate = tune(),
             tree_depth = tune())


# List of models
models <- list(lr_model = lr_model,
               dt_model = dt_model,
               rf_model = rf_model,
               xgb_model = xgb_model)

# Create workflow set
heart_workflows <- workflow_set(preproc = recipes,
                              models = models, 
                              cross = FALSE)

# Parallelization
registerDoMC(cores = 8) # 8 is to much, need some power for other task

# Fit models

heart_fit <- heart_workflows %>% 
  workflow_map("tune_grid",
               seed = 123,
               verbose = TRUE,
               metrics = metric_set(accuracy, roc_auc),
               resamples = heart_folds, 
               control = control_resamples(save_pred = TRUE,
                                           parallel_over = "everything",
                                           save_workflow = TRUE))
show_notes(.Last.tune.result)
str(heart_disease)
# Evaluation
# ------------------------------------------------------------------------------

# workflow
heart_wflow_down <- 
  workflow() %>% 
  add_model(dt_model) %>% 
  add_recipe(heart_recipe_down)
 
# heart_wflow_up <- 
#   workflow() %>% 
#   add_model(dt_model) %>% 
#   add_recipe(heart_recipe_up)

# show workflow
heart_wflow_down
heart_wflow_up