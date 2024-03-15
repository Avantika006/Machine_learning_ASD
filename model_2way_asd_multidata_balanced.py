#!/usr/bin/env python
# coding: utf-8

from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Loading the different datasets
df1 = pd.read_csv("basic_medical_screening-2023-07-21.csv")
df2 = pd.read_csv("background_history_child-2023-07-21.csv")
df3 = pd.read_csv("background_history_sibling-2023-07-21.csv")
df4 = pd.read_csv("individuals_registration-2023-07-21.csv", low_memory=False)

dict_mother = dict(zip(df2['family_sf_id'], df2['mother_highest_education']))
dict_father = dict(zip(df2['family_sf_id'], df2['father_highest_education']))
dict_income = dict(zip(df2['family_sf_id'], df2['annual_household_income']))

for idx, item in df3['family_sf_id'].items():
    if item in dict_mother:
        df3.loc[idx, 'mother_highest_education'] = dict_mother[item]
    if item in dict_father:
        df3.loc[idx, 'father_highest_education'] = dict_father[item]
    if item in dict_income:
        df3.loc[idx, 'annual_household_income'] = dict_income[item]

# Now, concatenate row-wise
final_df = pd.concat([df2, df3], ignore_index=True)

# Intersect with df1
final_df_with_df1 = pd.merge(final_df, df1, on='subject_sp_id', how='inner')

# Intersect with df4
merged_df = pd.merge(final_df_with_df1, df4, on='subject_sp_id', how='inner')

merged_df = merged_df.sample(frac=1.0, random_state=42)  # random_state for reproducibility

# Reset index (optional, but recommended for further analysis)
merged_df.reset_index(inplace=True)

# Create a new column 'prediction' based on ASD and ADHD statuses
prediction = []
for i in range(len(merged_df)):
    asd_status = str(merged_df['asd'][i]).lower() in ['true', 'true.']
    adhd_status = not pd.isna(merged_df['behav_adhd'][i])

    if asd_status :
        prediction.append(1)  # ASD present
    else:
        prediction.append(0)  # ASD absent

merged_df['prediction'] = prediction

# Separate the dataset into ASD-affected and unaffected individuals
merged_df_asd_affected = merged_df[merged_df['prediction'] == 1]
merged_df_unaffected = merged_df[merged_df['prediction'] == 0]

# Determine the number of ASD-affected individuals
num_asd_affected = len(merged_df_asd_affected)
num_asd_unaffected = len(merged_df_unaffected)

# Randomly sample an equal number of rows from the unaffected individuals subset
merged_df_affected_sampled = merged_df_asd_affected.sample(n=num_asd_unaffected, random_state=42)

# Concatenate the ASD-affected subset and the randomly sampled unaffected individuals subset
merged_df_balanced = pd.concat([merged_df_unaffected, merged_df_affected_sampled])

# Shuffle the rows of the final dataset
merged_df = merged_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop irrelevant columns
features = merged_df[[
    'sex', 'gest_age', 'eating_probs', 'feeding_dx',
    'med_cond_birth', 'birth_oth_calc',
    'med_cond_birth_def',
    'med_cond_growth', 'growth_oth_calc',
    'med_cond_neuro', 'med_cond_visaud',  'mother_highest_education', 'father_highest_education', 'annual_household_income',
    'smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
    'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos',
    'combined_phrases_age_mos', 'bladder_trained_age_mos', 'bowel_trained_age_mos', 'hand',
    'twin_mult_birth', 'num_asd_parents', 'num_asd_siblings'
]]

# Resetting indices after dropping irrelevant columns
features.reset_index(drop=True, inplace=True)

# Replace null values with 0 in all columns of df2
features = features.replace(np.float64('nan'), 0)
features["gest_age"] = features["gest_age"].replace(0,40)   #Replacing null values of gest age with 40

# Categorical columns
cat_col = ['sex', 'eating_probs', 'feeding_dx',
    'med_cond_birth', 'birth_oth_calc',
    'med_cond_birth_def',
    'med_cond_growth', 'growth_oth_calc',
    'med_cond_neuro', 'med_cond_visaud', 'mother_highest_education', 'father_highest_education',
    'hand',
    'twin_mult_birth','annual_household_income']
print('Categorical columns :',cat_col)

# Numerical columns
num_col = ['gest_age','smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
    'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos', 'combined_phrases_age_mos', 'bladder_trained_age_mos', 'bowel_trained_age_mos', 'num_asd_parents', 'num_asd_siblings']
print('Numerical columns :',num_col)

for col in cat_col:
    features[col] = features[col].astype(str)

for col in num_col:
    features[col] = features[col].astype(float)

#################################################################
###-----MODEL-----###\

# Separate features (X) and target variable (y)
X = features
y = merged_df['prediction']

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_col),
        ('cat', categorical_transformer, cat_col)])

mod = XGBClassifier()
#mod = svm.SVC(probability=True)
#mod = RandomForestClassifier(n_estimators=10)
#mod = LogisticRegression(max_iter=5000)
#mod = tree.DecisionTreeClassifier()

accuracy = []
sensitivity = []
specificity = []
f1 = []
precision = []

# K fold cross-validation and metric calculation
print("Performing 10 fold cross-validation and printing metrics for each fold...")
print("\n")
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterate over folds
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print("-----------------------", fold, "------------------------")
    print('CROSS VALIDATION NO-> ', fold)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Before fitting the data into the pipeline
    #print("Data types of columns before fitting into the pipeline:")
    #print(features.dtypes)

    # Fitting and transforming the training set
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transforming the testing set
    X_test_transformed = preprocessor.transform(X_test)

    mod.fit(X_train_transformed, y_train)
    y_pred = mod.predict(X_test_transformed)

    # Calculating the accuracy_score for each fold
    accuracy.append(accuracy_score(y_test, y_pred))

    # Calculating the sensitivity and specificity
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Adjust indices for 4 classes (0, 1, 2, 3)
    TP = conf_matrix[1:, 1:].sum()  # True Positives across all classes
    FP = conf_matrix[1:, 0].sum()  # False Positives
    FN = conf_matrix[0, 1:].sum()  # False Negatives
    TN = conf_matrix[0, 0]  # True Negatives

    sens = TP / (TP + FN)
    sensitivity.append(sens)
    spec = TN / (TN + FP)
    specificity.append(spec)

    # Calculating the F1 score
    f1.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculating the Precision
    precision.append(precision_score(y_test, y_pred, average='weighted'))

    ##TABLE PRINTING
    metrics_data = [
        ("Accuracy", accuracy[fold-1]),
        ("Sensitivity", sensitivity[fold-1]),
        ("Specificity", specificity[fold-1]),
        ("F1 score", f1[fold-1]),
        ("Precision", precision[fold-1]),
    ]

    # Print the table
    table_headers = ["Metric", "Value"]
    table = tabulate(metrics_data, headers=table_headers, tablefmt="grid")

    print(table)
    print("----------------------------------------------------------------------------------------")

##Calculation of mean and standard deviation of various metrics

# Accuracy
m_accuracy = np.mean(accuracy)
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