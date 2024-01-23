#!/usr/bin/env python
# coding: utf-8
#1. LINEAR REGRESSION USING CALIFORNIA HOUSING DATASET
# In[13]:


from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# load the California housing dataset and split it into train and test sets
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Fitting the model using the training sets
mod = LinearRegression()
mod.fit(X_train,y_train)

#Making predictions on the testing set
y_pred = mod.predict(X_test)

#Calculating the r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

#2. LOGISTIC REGRESSION USING BREAST CANCER DATASET
# In[16]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# load the California housing dataset and split it into train and test sets
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Fitting the model using the training sets
mod = LogisticRegression()
mod.fit(X_train,y_train)

#Making predictions on the testing set
y_pred = mod.predict(X_test)

#Calculating the r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)


# In[ ]:




