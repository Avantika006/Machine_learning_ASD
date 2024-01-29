#!/usr/bin/env python
# coding: utf-8

# RANDOM FORESTS CLASSIFIER USING BREAST CANCER DATASET - WITH STANDARDIZATION AND 10 FOLD CV
# In[16]:

from numpy import mean
from numpy import std
import numpy as np
from tabulate import tabulate
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import metrics


# Loading the Breast Cancer dataset and splitting it into train and test sets
X, y = load_breast_cancer(return_X_y=True)

# Standardization of the dataset and performing logistic regression
trans = StandardScaler()
mod = RandomForestClassifier(n_estimators=10)


#Declaring arrays for various metrics
#accuracy = []
num_tests = 11
accuracy = np.zeros(num_tests)
auc = np.zeros(num_tests)
sensitivity = np.zeros(num_tests)
specificity = np.zeros(num_tests)
f1 = np.zeros(num_tests)
precision = np.zeros(num_tests)

# K fold cross-validation and accuracy prediction
print("Performing 10 fold cross validation and printing metrics for each fold...")
print("\n")
kf = StratifiedKFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print("-----------------------", fold,"------------------------")
    print('CROSS VALIDATION NO-> ', fold)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Standardization
    trans.fit(X_train)
    x_scaled = trans.transform(X_test)

    # Making predictions on the testing set
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    # Calculating the accuracy_score for each fold
    accuracy[fold] = accuracy_score(y_test, y_pred)

    # Calculating the Area under Curve (AUC)
    y_pred_proba = mod.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)  # Calculate the fpr, tpr, and threshold
    auc[fold] = metrics.auc(fpr, tpr)  # Calculate the AUC

    # Calculating the sensitivity and specificity
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    TP = conf_matrix[1, 1]  # True Positive (TP)
    FP = conf_matrix[0, 1]  # False Positive (FP)
    FN = conf_matrix[1, 0]  # False Negative (FN)
    TN = conf_matrix[0, 0]  # True Negative (TN)

    sensitivity[fold] = TP / (TP + FN)  # Sensitivity (True Positive Rate or Recall)

    specificity[fold] = TN / (TN + FP)  # Specificity (True Negative Rate)

    # Calculating the F1 score
    f1[fold] = f1_score(y_test, y_pred)

    # Calculating the Precision
    precision[fold] = precision_score(y_test, y_pred)

    ##TABLE PRINTING
    metrics_data = [
        ("Accuracy", accuracy[fold]),
        ("AUC", auc[fold]),
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

#Accuracy
m_accuracy  = mean(accuracy)
std_accuracy = std(accuracy)

#AUC
m_auc = mean(auc)
std_auc = std(auc)

#Sensitivity
m_sensitivity = mean(sensitivity)
std_sensitivity = std(sensitivity)

#Specificity
m_specificity = mean(specificity)
std_specificity = std(specificity)

#F1 score
m_f1 = mean(f1)
std_f1 = std(f1)

#Precision
m_precision = mean(precision)
std_precision = std(precision)

##TABLE PRINTING
metrics_data = [
    ("Accuracy", m_accuracy, std_accuracy),
    ("AUC", m_auc, std_auc),
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