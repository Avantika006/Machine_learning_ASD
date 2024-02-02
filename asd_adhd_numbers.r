df <- read.csv("basic_medical_screening-2023-07-21.csv")

# Initializing counts
asd_only = 0
adhd_only = 0
both_asd_adhd = 0
unaffected = 0

# Looping through each row
for (i in 1:nrow(df)) {
  
  # Checking ASD and ADHD statuses
  asd_status <- df$asd[i] %in% c("TRUE", "True")
  adhd_status <- !is.na(df$behav_adhd[i])
  
  # Updating counts based on conditions
  if (asd_status & adhd_status) {
    both_asd_adhd = both_asd_adhd + 1
  } else if (asd_status) {
    asd_only = asd_only + 1
  } else if (adhd_status) {
    adhd_only = adhd_only + 1
  } else {
    unaffected = unaffected + 1
  }
}

# Display results
cat(paste("Number of individuals affected by ASD only:", asd_only), "\n")
cat(paste("Number of individuals affected by ADHD only:", adhd_only), "\n")
cat(paste("Number of individuals affected by both ASD and ADHD:", both_asd_adhd), "\n")
cat(paste("Number of individuals unaffected by any of the 5 disorders:", unaffected), "\n")
