#!/usr/bin/env python
# coding: utf-8

from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Loading the Basic Medical Screening dataset
df = pd.read_csv("basic_medical_screening-2023-07-21.csv")

# Create a new column 'prediction' based on ASD and ADHD statuses
prediction = []
for i in range(len(df)):
    asd_status = str(df['asd'][i]).lower() in ['true', 'true.']
    adhd_status = not pd.isna(df['behav_adhd'][i])

    if asd_status and adhd_status:
        prediction.append(3)  # Both ASD and ADHD
    elif asd_status:
        prediction.append(1)  # ASD only
    elif adhd_status:
        prediction.append(2)  # ADHD only
    else:
        prediction.append(0)  # Neither ASD nor ADHD

df['prediction'] = prediction
print(df.info())
null_counts = df.isnull().sum()
#print("Number of null values for each column:")
#print(null_counts)

# Create a DataFrame to store the null counts
null_counts_df = pd.DataFrame({'Column Name': null_counts.index, 'Null Count': null_counts.values})

# Write the DataFrame to an Excel file
null_counts_df.to_excel('null_counts.xlsx', index=False)

# Drop irrelevant columns
features = df.drop(columns=['subject_sp_id', 'respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id',
                            'current_depend_adult', 'attn_behav', 'birth_def_cns', 'birth_def_bone', 'birth_def_fac',
                            'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen', 'dev_lang', 'gen_test',
                            'med_cond_birth', 'med_cond_birth_def', 'med_cond_growth', 'med_cond_neuro', 'med_cond_visaud',
                            'mood_ocd', 'prediction'])

# Separate features (X) and target variable (y)
X = features
y = df['prediction']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=categorical_columns)

# Convert y to integer for StratifiedKFold
y = y.astype(int)

# Standardization of the dataset and performing XGBoost classification
trans = StandardScaler()
mod = XGBClassifier(objective='multi:softmax', num_class=4)  # Change to 4 classes

# Declaring arrays for various metrics
num_tests = 11
accuracy = np.zeros(num_tests)
sensitivity = np.zeros(num_tests)
specificity = np.zeros(num_tests)
f1 = np.zeros(num_tests)
precision = np.zeros(num_tests)

# K fold cross-validation and accuracy prediction
print("Performing 10 fold cross-validation and printing metrics for each fold...")
print("\n")
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print("-----------------------", fold, "------------------------")
    print('CROSS VALIDATION NO-> ', fold)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardization
    trans.fit(X_train, y_train)
    y_scaled = trans.transform(X_test)

    # Making predictions on the testing set
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    # Calculating the accuracy_score for each fold
    accuracy[fold] = accuracy_score(y_test, y_pred)

    # Calculating the sensitivity and specificity
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Adjust indices for 4 classes (0, 1, 2, 3)
    TP = conf_matrix[1:, 1:].sum()  # True Positives across all classes
    FP = conf_matrix[1:, 0].sum()  # False Positives
    FN = conf_matrix[0, 1:].sum()  # False Negatives
    TN = conf_matrix[0, 0]  # True Negatives

    sensitivity[fold] = TP / (TP + FN)
    specificity[fold] = TN / (TN + FP)

    # Calculating the F1 score
    f1[fold] = f1_score(y_test, y_pred, average='weighted')

    # Calculating the Precision
    precision[fold] = precision_score(y_test, y_pred, average='weighted')

    ##TABLE PRINTING
    metrics_data = [
        ("Accuracy", accuracy[fold]),
        ("Sensitivity", sensitivity[fold]),
        ("Specificity", specificity[fold]),
        ("F1 score", f1[fold]),
        ("Precision", precision[fold]),
    ]

    # Print the table
    table_headers = ["Metric", "Value"]
    table = tabulate(metrics_data, headers=table_headers, tablefmt="grid")

    print(table)
    print("----------------------------------------------------------------------------------------")

##Calculation of mean and standard deviation of various metrics

# Accuracy
m_accuracy = mean(accuracy)
std_accuracy = std(accuracy)

# Sensitivity
m_sensitivity = mean(sensitivity)
std_sensitivity = std(sensitivity)

# Specificity
m_specificity = mean(specificity)
std_specificity = std(specificity)

# F1 score
m_f1 = mean(f1)
std_f1 = std(f1)

# Precision
m_precision = mean(precision)
std_precision = std(precision)

##TABLE PRINTING
metrics_data = [
    ("Accuracy", m_accuracy, std_accuracy),
    ("Sensitivity", m_sensitivity, std_sensitivity),
    ("Specificity", m_specificity, std_specificity),
    ("F1 score", m_f1, std_f1),
    ("Precision", m_precision, std_precision),
]

# Print the table
table_headers = ["Metric", "Mean", "Standard Deviation"]
table = tabulate(metrics_data, headers=table_headers, tablefmt="grid")

print("MEAN AND STANDARD DEVIATION OF VARIOUS METRICS")
print(table)
