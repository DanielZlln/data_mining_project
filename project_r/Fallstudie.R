install.packages("DataExplorer")
install.packages("psych")
library(dplyr)
library(DataExplorer)
library(psych)


### Daten einlesen

load("heart_disease.RData")

View(heart_disease)
summary(heart_disease)
# Kaum NA -> löschen der Zeilen

str(heart_disease)
# Hoher_Blutdruck max. 630 -> unlogisch muss 1 / 0
# Hoher_Cholspiegel max. 867 -> unlogisch muss 1 / 0
# BMI max. 98 sinnvoll? Bei 180cm über 300kg
# Rauchen max. 833 -> unlogisch muss 1 / 0
# Phys_Aktivität Wert 891 -> unlogisch muss 1 / 0
# ALLE ZEILEN LOESCHEN MIT DEN OBEREN WERTEN DA SO WENIG

introduce(heart_disease)

heart_disease %>% 
  filter(Hoher_Cholspiegel > 1 | Hoher_Blutdruck > 1 |
           Rauchen > 1 | Phys_Aktivität > 1) %>% 
  select(Hoher_Blutdruck, Hoher_Cholspiegel, Rauchen, Phys_Aktivität) %>% 
  summarise(anzahl = n())
# Insgeasmt 16 Zeilen mit falschen Werten

# Infarkt
mean(heart_disease$Herzinfarkt == "Infarkt") 
# 9.5% der Patienten haben ein Infarkt -> Balancing

# Geschlecht
heart_disease %>% 
  group_by(Geschlecht) %>% 
  summarise(anzahl = n())
# Mehr Frauen als Männer

# Einkommen
mode(heart_disease$Einkommen) # Einkommen ist "character"

# Schulabschluss
heart_disease %>% 
  distinct(Bildung)

heart_disease %>% 
  group_by(Bildung) %>% 
  summarise(anzahl = n()) %>% 
  arrange(desc(anzahl))

# Spalte BMI 
boxplot(heart_disease$BMI~heart_disease$Alter) # Wieso so viele Ausreiser nach oben?
mean(heart_disease$BMI>30, na.rm = T) 

h = hist(heart_disease$BMI, breaks = 100)
ccat = cut(h$breaks, c(-Inf, 30, Inf))
plot(h, col=c("white","red")[ccat])


# Daten bereinigen
for (i in 1:ncol(heart_disease)) {
  if (is.numeric(heart_disease[[i]])) {
    if (mean(heart_disease[[i]], na.rm = T) < 1 
        & max(heart_disease[[i]], na.rm = T) > 2) {
      for (j in length(heart_disease[,i])) {
        if (is.na(heart_disease[j,i])) {
          next
        }
        print(heart_disease[j,i])
        if (heart_disease[j,i] > 1) {
          print("yes")
        }
      }
    }
  }
}

summary(heart_disease)
str(heart_disease)
mean(heart_disease$Rauchen, na.rm = T)
