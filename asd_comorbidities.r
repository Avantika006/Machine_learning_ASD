
df <- read.csv("basic_medical_screening-2023-07-21.csv")
  
#ASD prediction
i = ""
asd = 0
no_asd = 0 
for (i in df$asd){
  if (i == "TRUE" | i == "True"){
    asd = asd + 1
  }
  else
    if(i == "FALSE" | i == "False"){
      no_asd = no_asd + 1
    }
}

num_indiv <- nrow(df)

#cat("File:", file_path, "\n")

#ADHD prediction
adhd = 0
no_adhd = 0

for( m in df$behav_adhd){
  if(!is.na(m) == TRUE){
    no_adhd = no_adhd + 1
  }
  else{
      adhd = adhd + 1
  }
}


#ODD prediction
n = 0
odd = 0
no_odd = 0

for( n in df$behav_odd){
  if(!is.na(n) == TRUE){
    no_odd = no_odd + 1
  }
  else{
    odd = odd + 1
  }
}


#OCD prediction
p = 0
ocd = 0
no_ocd = 0

for( p in df$mood_ocd){
  if(!is.na(p) == TRUE){
    no_ocd = no_ocd + 1
  }
  else{
    ocd = ocd + 1
  }
}


#Schizophrenia prediction
q = 0
sciz = 0
no_sciz = 0

for( q in df$schiz){
  if(!is.na(q) == TRUE){
    no_sciz = no_sciz + 1
  }
  else{
    sciz = sciz + 1
  }
}


cat(paste("Total number of individuals:",num_indiv), "\n\n")

cat(paste("Disease -> ASD"))
cat(paste("Number of ASD affected individuals:", asd), "\n")
cat(paste("Number of non-affected individuals:", no_asd), "\n\n")

cat(paste("Disease -> ADHD"))
cat(paste("Number of ADHD affected individuals:", adhd), "\n")
cat(paste("Number of non-affected individuals:", no_adhd), "\n\n")

cat(paste("Disease -> ODD"))
cat(paste("Number of ODD affected individuals:", odd), "\n")
cat(paste("Number of non-affected individuals:", no_odd), "\n\n")

cat(paste("Disease -> OCD"))
cat(paste("Number of OCD affected individuals:", ocd), "\n")
cat(paste("Number of non-affected individuals:", no_ocd), "\n\n")

cat(paste("Disease -> Schizophrenia"))
cat(paste("Number of Schizophrenia affected individuals:", sciz), "\n")
cat(paste("Number of non-affected individuals:", no_sciz), "\n\n")
