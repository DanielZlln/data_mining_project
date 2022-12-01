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

#max(heart_disease$Hoher_Cholspiegel > 1, na.rm = T)
