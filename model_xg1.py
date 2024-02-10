#!/usr/bin/env python
# coding: utf-8

from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
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

# Filter rows where prediction is not 0
df2 = df[df['prediction'] != 0].copy()

# Drop irrelevant columns
features = df2.drop(columns=['subject_sp_id', 'respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id',
                            'current_depend_adult', 'attn_behav', 'birth_def_cns', 'birth_def_bone', 'birth_def_fac',
                            'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen', 'dev_lang', 'gen_test',
                            'med_cond_birth', 'med_cond_birth_def', 'med_cond_growth', 'med_cond_neuro', 'med_cond_visaud',
                            'mood_ocd', 'prediction','behav_conduct','behav_intermitt_explos','behav_odd','basic_medical_measure_validity_flag',
                            'birth_def_bone_club','birth_def_bone_miss','birth_def_bone_polydact','birth_def_bone_spine','birth_def_cleft_lip',
                            'birth_def_cleft_palate','birth_def_cns_myelo','birth_def_gi_esoph_atres','birth_def_gi_hirschprung','birth_def_gi_intest_malrot',
                            'birth_def_gi_pylor_sten', 'birth_def_thorac_heart', 'birth_def_thorac_lung', 'birth_def_urogen_hypospad', 'birth_def_urogen_renal',
                            'birth_def_urogen_renal_agen', 'birth_def_urogen_uter_agen','birth_def_oth_calc', 'birth_ivh', 'birth_oth_calc','etoh_subst',
                            'gen_dx_oth_calc_self_report','gen_test_cgh_cma','gen_test_chrom_karyo','gen_test_ep','gen_test_fish_angel','gen_test_fish_digeorge',
                            'gen_test_fish_williams','gen_test_fish_oth','gen_test_frax','gen_test_id','gen_test_mecp2','gen_test_nf1','gen_test_noonan','gen_test_pten',
                            'gen_test_tsc','gen_test_unknown','gen_test_wes','gen_test_wgs','gen_test_oth_calc','growth_low_wt','growth_macroceph',
                            'growth_microceph','growth_obes','growth_short','growth_oth_calc','prev_study_calc','eval_year','neuro_inf','neuro_lead',
                            'neuro_sz','neuro_tbi','neuro_oth_calc','pers_dis','prev_study_oth_calc','psych_oth_calc','schiz','visaud_blind',
                            'visaud_catar','visaud_deaf','visaud_strab','tics']).copy()

# Replace null values with 0 in all columns of df2
df2.fillna(0, inplace=True)

#Check for null values
null_counts = df2.isnull().sum()

# Create a DataFrame to store the null counts
null_counts_df = pd.DataFrame({'Column Name': null_counts.index, 'Null Count': null_counts.values})

# Write the DataFrame to an Excel file
null_counts_df.to_excel('null_counts_new.xlsx', index=False)

#Viewing dimensions of features
num_columns = features.shape[1]
print("Number of features post dropping: ",num_columns)

# Categorical columns
cat_col = [col for col in df2.columns if df2[col].dtype == 'object' and col in features]
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in df2.columns if df2[col].dtype != 'object' and col in features]
print('Numerical columns :',num_col)

#Finding number of unique values in each of the categorical columns
print(df2[cat_col].nunique())

#################################################################
###-----MODEL-----###\

# Separate features (X) and target variable (y)
X = features
y = df2['prediction']

# Resetting indices after dropping irrelevant columns
features.reset_index(drop=True, inplace=True)

# Use LabelEncoder to encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_col),
        ('cat', categorical_transformer, cat_col)])

mod = XGBClassifier(objective='multi:softmax', num_class=3)

#print(y.unique())

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
    y_train, y_test = y[train_index], y[test_index]

    # Fitting and transforming the training set
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transforming the testing set
    X_test_transformed = preprocessor.transform(X_test)

    mod.fit(X_train_transformed, y_train)
    y_pred = mod.predict(X_test_transformed)

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
