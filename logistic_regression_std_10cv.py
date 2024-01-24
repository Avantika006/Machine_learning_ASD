#!/usr/bin/env python
# coding: utf-8

# LOGISTIC REGRESSION USING BREAST CANCER DATASET - WITH STANDARDIZATION AND 10 FOLD CV
# In[16]:

from numpy import mean
from numpy import std
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import metrics


# Loading the Breast Cancer dataset and splitting it into train and test sets
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Standardization of the dataset and performing logistic regression - pipeline
trans = StandardScaler()
mod = LogisticRegression()
pipeline = Pipeline(steps=[('t', trans), ('m', mod)])

#Evaluating the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# reporting the pipeline performance
print('Accuracy of pipeline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#Model fitting
pipeline.fit(X_train,y_train)

#Making predictions on the testing set
y_pred = pipeline.predict(X_test)


#Calculating the accuracy_score , fraction and count
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score - default:", accuracy)    #fraction
accuracy_no_norm = accuracy_score(y_test, y_pred, normalize=False)
print("Accuracy Score - no normalization:", accuracy_no_norm)   #count


# Calculating the Area under Curve (AUC)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)  # Calculate the fpr, tpr, and threshold
auc = metrics.auc(fpr, tpr) # Calculate the AUC
print("AUC:", auc)


#Calculating the sensitivity and specificity
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[1, 1]  # True Positive (TP)
FP = conf_matrix[0, 1]  # False Positive (FP)
FN = conf_matrix[1, 0]  # False Negative (FN)
TN = conf_matrix[0, 0]  # True Negative (TN)

sensitivity = TP / (TP + FN)    # Sensitivity (True Positive Rate or Recall)
print("Sensitivity:", sensitivity)

specificity = TN / (TN + FP)    # Specificity (True Negative Rate)
print("Specificity:", specificity)


#Calculating the F1 score
f1 = f1_score(y_test,y_pred)
print("F1 score:", f1)


#Calculating the Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)


#EXTRA ---> Printing the classification report
print(classification_report(y_test,y_pred))