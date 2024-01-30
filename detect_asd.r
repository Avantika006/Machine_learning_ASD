process_csv <- function(file_path) {
  df <- read.csv(file_path)
  
  i = ""
  asd = 0
  no_asd = 0 
  no_data = 0 
  for (i in df$asd){
    if (i == "TRUE" | i == "True"){
      asd = asd + 1
    }
    else
      if(i == "FALSE" | i == "False"){
        no_asd = no_asd + 1
      }
    else
    {
      no_data = no_data + 1
    }
  }
  
  num_indiv <- nrow(df)
  
  cat("File:", file_path, "\n")
  cat(paste("Total number of individuals:",num_indiv), "\n")
  cat(paste("Number of ASD affected individuals:", asd), "\n")
  cat(paste("Number of non-affected individuals:", no_asd), "\n")
  cat(paste("Number of individuals with no data available:", no_data), "\n\n")
}

# List of CSV files
files <- c("core_descriptive_variables-2023-07-21.csv", "basic_medical_screening-2023-07-21.csv", "cbcl_1_5-2023-07-21.csv","cbcl_6_18-2023-07-21.csv","dcdq-2023-07-21.csv","individuals_registration-2023-07-21.csv","iq-2023-07-21.csv","rbsr-2023-07-21.csv","roles_2023-07-17.csv","scq-2023-07-21.csv","srs-2_adult_self-2023-07-21.csv","vineland-3-2023-07-21.csv")

# Process each file
for (file in files) {
  process_csv(file)
}
