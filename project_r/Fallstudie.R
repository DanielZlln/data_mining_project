#------------------------------------------------------------------------------
# Data Mining 
# Praeventionsoptimierung durch Vorhersage von Herzinfarkten 
# Daniel Zellner 1153271 
# Nele Saathoff 1153284
#------------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Pakete und Datensatz laden
#------------------------------------------------------------------------------


# Pakete installieren
# install.packages("DataExplorer")
# install.packages("psych")
# install.packages("reticulate")
# devtools::install_github("tidymodels/tune")
# install.packages("doMC")
# install.packages("ranger")
# install.packages("xgboost")
# install.packages("caret")


# library laden
library(lattice)
library(glmnet)
library(vip)
library(caret)
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


#-----------------------------------------------------------------------------
# Datenaufbereitung, Ueberblick verschaffen
#------------------------------------------------------------------------------

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

# wie viele falsche Werte liegen im Datensatz?
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
# 2 Zeilen "unbekannt" loeschen

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

heart_disease %>% 
  group_by(Einkommen) %>% 
  summarise(anzahl = n())

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

# Zeilen loeschen die falsche Werte enthalten
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
  filter(Einkommen != "unbekannt")

heart_disease <- heart_disease %>% 
  filter(complete.cases(.))

# Spalte BMI 
boxplot(heart_disease$BMI~heart_disease$Alter) # Wieso so viele Ausreiser nach oben?
mean(heart_disease$BMI>30, na.rm = T) 

# Histogramm erstellen
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

#-----------------------------------------------------------------------------
# Datenaufbereitung, Ueberblick verschaffen
#------------------------------------------------------------------------------

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

# Create recipe
heart_recipe_down <- 
  recipe(Herzinfarkt ~., data = heart_train) %>% 
  step_downsample(Herzinfarkt, under_ratio = 3) %>%
  step_normalize(all_numeric_predictors())

# heart_recipe_up <-
#   recipe(Herzinfarkt ~., 
#          data = heart_train) %>%
#   step_smotenc(Herzinfarkt, 
#                over_ratio = 1) %>% 
#   step_normalize(all_nominal_predictors())

# Create second recipe with dummy 
heart_recipe_down_num <- heart_recipe_down %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# heart_recipe_up_num <- heart_recipe_up %>% 
#   step_dummy(all_nominal_predictors(), 
#              one_hot = TRUE)

recipes <- list(lr_recipe = heart_recipe_down)
#lr_tune_recipe = heart_recipe_down_num)
#dt_recipe = heart_recipe_down)
#rf_recipe = heart_recipe_down,
#xgb_recipe = heart_recipe_down_num)

#-------------------------------------------------------------------------------
# Resampling
#-------------------------------------------------------------------------------

# Cross Validation
set.seed(123)
heart_folds <- vfold_cv(heart_train)
heart_folds


#------------------------------------------------------------------------------
# Modeling
#------------------------------------------------------------------------------


# Logistic Regression
lr_model <- 
  logistic_reg(mode = "classification", 
               engine = "glm")

# Logistic tune, keine verbesserung!
lr_mode_tune <- 
  logistic_reg(engine = "glmnet", 
               mode = "classification",
               penalty = tune(),
               mixture = tune())

# Decision Tree
dt_model <- 
  decision_tree(mode = "classification",
                engine = "rpart",
                cost_complexity = tune(),
                min_n = tune(),
                tree_depth = tune())

# Random Forest
rf_model <- 
  rand_forest(mode = "classification",
              engine = "ranger",
              trees = tune(),
              min_n = tune())
#mtry = tune()) wird nicht erkannt

# XGBoost
xgb_model <- 
  boost_tree(mode = "classification",
             engine = "xgboost",
             trees = 1000, 
             tree_depth = tune(), 
             min_n = tune(), 
             loss_reduction = tune(), 
             sample_size = tune(), 
             mtry = tune(),         
             learn_rate = tune())
# don´t start this one!! :D



# List of models
models <- list(lr_model = lr_model)
#lr_model_tune = lr_mode_tune)
#dt_model = dt_model,
#rf_model = rf_model)
#xgb_model = xgb_model)

# Create workflow set
heart_workflows <- workflow_set(preproc = recipes,
                                models = models, 
                                cross = FALSE)

# Parallelization
registerDoMC(cores = 6) # 8 is to much, need some power for other task

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


# Logistic is the best one

lr_form_fit <- 
  lr_model %>% 
  fit(Herzinfarkt~., data = heart_train)

lr_form_fit %>% extract_fit_engine() %>% summary()

# Overfitting = zu komplexes Modell
# die optimale Modellkomplexität finden

# Speichern Vorhersagen Trainingsdaten
pred_train <-
  heart_train %>% select(Herzinfarkt) %>%
  bind_cols(predicted = predict(lr_form_fit,
                                new_data = heart_train,
                                type = "prob")) %>%
  mutate(dataset = "train")

# Speichern Vorhersagen Testdaten
pred_test <-
  heart_test %>% select(Herzinfarkt) %>%
  bind_cols(predicted = predict(lr_form_fit,
                                new_data = heart_test,
                                type = "prob")) %>%
  mutate(dataset = "test")

# Kombinieren von Test- und Trainingsdaten
pred <- bind_rows(pred_train, pred_test)

# Plot beide ROCs in einer Abbildung
pred %>%
  group_by(dataset) %>%
  roc_curve(truth = Herzinfarkt, estimate = `.pred_Infarkt`) %>%
  autoplot()


# falls die Kurven stark voneinander abweichen, muss man von overfitting 
# ausgehen - hier weichen die Kurven nicht voneinander ab

#------------------------------------------------------------------------------
# Evaluation
#------------------------------------------------------------------------------
collection <- collect_metrics(heart_fit, summarize = F)
collection

# Accuracy und ROC AUC darstellen 
autoplot(heart_fit,
         rank_metric = "roc_auc",
         metrich = "roc_auc",
         select_best = T)

heart_fit %>% rank_results(rank_metric = "roc_auc",
                           select_best = T) %>% 
  filter(.metric == "roc_auc")

best_result <- 
  heart_fit %>% 
  extract_workflow_set_result("lr_recipe_lr_model") %>% 
  select_best(metric = "roc_auc")

best_result

best_test_results <- 
  heart_fit %>% 
  extract_workflow("lr_recipe_lr_model") %>% 
  finalize_workflow(best_result) %>% 
  last_fit(split = heart_split)

best_test_results %>% 
  collect_metrics()


#-------------------------------------------------------------------------------
# Prediction
#-------------------------------------------------------------------------------

best_test_results$.predictions

# Prediction on test

heart_test %>% 
  select(Herzinfarkt) %>% 
  bind_cols(
    predicted = predict(extract_fit_parsnip(best_test_results),
                        new_data = heart_test,
                        type = "prob")) %>% 
  roc_auc(truth = Herzinfarkt, estimate = `.pred_Infarkt`)

# Matrix erstellen 
kosten_ertaege <- matrix(c(5000, -5000, -1000, 0), nrow = 2, ncol = 2)
neu <- 0
best_threshold <- 0
for (i in seq(0,1, by = 0.01)) {
  
  conf_erg <- heart_test %>% 
    select(Herzinfarkt) %>% 
    bind_cols(
      predicted = predict(extract_fit_parsnip(best_test_results),
                          new_data = heart_test,
                          type = "prob")) %>% 
    mutate(.pred_class = make_two_class_pred(`.pred_Infarkt`,
                                             levels(Herzinfarkt),
                                             threshold = i)) %>% 
    conf_mat(estimate = .pred_class, truth = Herzinfarkt)
  aktuell <- sum(conf_erg$table * kosten_ertaege)
  
  if (aktuell > neu) {
    neu <- aktuell
    best_threshold <- i
  }
}

conf_erg <- heart_test %>% 
  select(Herzinfarkt) %>% 
  bind_cols(
    predicted = predict(extract_fit_parsnip(best_test_results),
                        new_data = heart_test,
                        type = "prob")) %>% 
  mutate(.pred_class = make_two_class_pred(`.pred_Infarkt`,
                                           levels(Herzinfarkt),
                                           threshold = .47)) %>% 
  conf_mat(estimate = .pred_class, truth = Herzinfarkt)
sum(conf_erg$table*kosten_ertaege)



# Welche Variablen haben groessten einfluss

best_result_fitted <- best_test_results$.workflow[[1]]

lr_explain <- explain_tidymodels(
  best_result_fitted,
  data = dplyr::select(heart_disease, -Herzinfarkt),
  y = abs(as.integer(heart_disease$Herzinfarkt)-1),
  predict_function = function(lr_model, newdata)
    pull(predict(lr_model,newdata, type = "prob")[".pred_Infarkt"]),
  verbose = F
)

lr_vip <- lr_explain %>% 
  model_parts(type = "variable_importance")

head(lr_vip)

plot(lr_vip)

# BMI hat den groessten Einlfuss und kann mit prevention verbessert werden


# heart disease result
load("heart_disease_result.RData")
str(heart_disease_result)
summary(heart_disease_result)

# Heart result so aufbereiten wie den alten

heart_disease_result <- heart_disease_result %>% 
  filter(Alter != "unbekannt")

heart_disease_result <- heart_disease_result %>% 
  filter(Geschlecht != "unbekannt")

heart_disease_result <- heart_disease_result %>% 
  filter(Hoher_Blutdruck <= 1)

heart_disease_result <- heart_disease_result %>% 
  filter(Hoher_Cholspiegel <= 1)

heart_disease_result <- heart_disease_result %>% 
  filter(Rauchen <= 1)

heart_disease_result <- heart_disease_result %>% 
  filter(Phys_Aktivität <= 1)

heart_disease_result <- heart_disease_result %>% 
  filter(Bildung != "unbekannt")

heart_disease_result <- heart_disease_result %>% 
  filter(Einkommen != "unbekannt")

heart_disease_result <- heart_disease_result %>% 
  filter(complete.cases(.))


# Vorhersage an neuen Datensatz
heart_disease_result$predicted <- predict(lr_form_fit, 
                                          heart_disease_result)

# Bei wie vielen wir ein Infarkt vorhergesagt
table(heart_disease_result$predicted)

# Speichern RDATA
save(heart_disease_result, file = "vorhersagen.RDATA")
