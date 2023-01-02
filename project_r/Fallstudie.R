# Pakete installieren
install.packages("DataExplorer")
install.packages("psych")
install.packages("reticulate")

# library laden
library(dplyr)
library(DataExplorer)
library(psych)
library(modeldata)
library(tidymodels)
tidymodels_prefer()
library(rpart)
library(rpart.plot)
library(probably)
library(tidyverse)
library(reticulate)

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

# Zeilen löschen die falsche Werte enthalten
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
  filter(complete.cases(.))

# Infarkt
mean(heart_disease$Herzinfarkt == "Infarkt") 
# 9.5% der Patienten haben ein Infarkt -> Balancing

# Geschlecht
heart_disease %>% 
  group_by(Geschlecht) %>% 
  summarise(anzahl = n())
# Mehr Frauen als Männer
# Geschlecht = "unbekannt" löschen da nur 3 Einträge

heart_disease <- heart_disease %>% 
  filter(Geschlecht != "unbekannt")

# Einkommen
mode(heart_disease$Einkommen) # Einkommen ist "character"

heart_disease %>% 
  group_by(Einkommen) %>% 
  summarise(anzahl = n()) %>% 
  arrange(desc(anzahl))

heart_disease <- heart_disease %>% 
  filter(Einkommen != "unbekannt")

# Schulabschluss
heart_disease %>% 
  group_by(Bildung) %>% 
  summarise(anzahl = n()) %>% 
  arrange(desc(anzahl))

# Bildung = unbekannt, löschen da nur 4 Einträge

heart_disease <- heart_disease %>% 
  filter(Bildung != "unbekannt")

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


# Correlation test
cor.test(x = heart_disease$Phys_Gesundheit,
         y = heart_disease$Ment_Gesundheit)

table(Gesundheit = heart_disease$Allg_Gesundheit,
      Aktivität = cut(heart_disease$Phys_Gesundheit + 
                        heart_disease$Ment_Gesundheit,
                      breaks = c(0,10,20,30,40,Inf),
                      dig.lab = 5,
                      include.lowest = T))
# Gesundheitsstatus gut -> mehr kleine Werte



#######
# Neue Spalten bilden?

lr_infarkt <- glm(Herzinfarkt ~ .,
                  data = heart_disease)

#######
set.seed(12345)
heart_split <- initial_split(heart_disease,
                             prop = 0.75,
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

# recipe prep
str(heart_test)

heart_train_prep <- 
  recipe(Herzinfarkt ~ .,
         data = heart_train) %>% 
  step_naomit(everything()) %>% 
  step_downsample(Herzinfarkt) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.5) %>%
  step_zv(all_numeric_predictors())


heart_train_prep 

heart_train_prep <- heart_train_prep %>% 
  prep() %>% 
  bake(new_data = NULL)

heart_train_prep %>% count(Herzinfarkt) %>% mutate(prop = n/sum(n))
heart_train_prep %>% distinct(Einkommen)

# Logistische Regression mit Parsnice
lr_model <- 
  logistic_reg() %>% 
  set_engine("glm")

lr_form_fit <- 
  lr_model %>% 
  fit(Herzinfarkt ~ ., data = heart_train_prep)

lr_form_fit %>% extract_fit_engine() %>% summary()
# Frau wird als Intercept gesetzt


##### Decision Tree
# Generate Model with Parsnip
# heart_train_red <- recipe(Herzinfarkt ~ ., data = heart_train) %>% 
#   step_downsample(Herzinfarkt) %>% 
#   step_dummy(all_nominal(), -all_nominal()) %>% 
#   step_zv(all_numeric()) %>% 
#   step_normalize(all_numeric()) %>% 
#   prep()
# 
# heart_train_red

dt_model <- 
  decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode(mode = "classification") %>% 
  set_args(parms = list(split = "information"))

dt_model_fit <- 
  dt_model %>% 
  fit(Herzinfarkt ~ ., data = heart_train_prep)

dt_model_fit %>% extract_fit_engine() %>% rpart.plot()


# Gini-Index
# rpart.plot(dt_model,
#            main = "Gini-Index")
# 
# dt_model_ig <- rpart(Herzinfarkt ~ ., data = juice(heart_train_red),
#                      parms = list(split = "information"))
# 
# rpart.plot(dt_model_ig,
#            main = "Information Gain Ratio")

######

# Prediction
predict(dt_model_fit, new_data = heart_train)
predict(dt_model_fit, new_data = heart_train, type = "prob")


heart_train$pred <- predict(dt_model_fit, new_data = heart_train)

heart_test %>% 
  select(1,23)

heart_train %>% select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit, new_data = heart_train)) %>% 
  conf_mat(truth = Herzinfarkt, estimate = .pred_class)
# Sehr oft Infarkt vorhergesagt, obwohl keiner passiert
# entstehen kosten von 5000 -> optimieren

class_metrics <- metric_set(accuracy, sens, spec)

heart_train %>% 
  select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit, new_data = heart_train)) %>% 
  class_metrics(truth = Herzinfarkt, estimate = .pred_class)

# accuracy: 0.681
# sens: 0.8
# spec: 0.669

heart_train %>% select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit,
                                new_data = heart_train,
                                type = "prob")) %>% 
  mutate(.pred_class = make_two_class_pred(`.pred_Infarkt`,
                                           levels(Herzinfarkt),
                                           threshold = .25)) %>% 
  conf_mat(truth = Herzinfarkt, estimate = .pred_class)

# ROC curve
heart_train %>% 
  select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit,
                                new_data = heart_train,
                                type = "prob")) %>% 
  roc_curve(truth = Herzinfarkt, estimate = `.pred_Infarkt`) %>% 
  autoplot()

heart_train %>% 
  select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit,
                                new_data = heart_train,
                                type = "prob")) %>% 
  roc_auc(truth = Herzinfarkt, estimate = `.pred_Infarkt`) 

# Lift curve
heart_train %>% 
  select(Herzinfarkt) %>% 
  bind_cols(predicted = predict(dt_model_fit,
                                new_data = heart_train,
                                type = "prob")) %>% 
  lift_curve(truth = Herzinfarkt, estimate = `.pred_Infarkt`) %>% 
  autoplot()
