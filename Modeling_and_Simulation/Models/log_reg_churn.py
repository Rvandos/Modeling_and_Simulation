#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# objective: train the churn dataset on a logistic regression model, with default param settings, balanced class weight,
# param includes use only features with coefficient > certain threshold, test size in train-test split
# also including specifying random state just in case
# the function returns: the model, the high coefficent features, and their coefficient
def log_reg_churn(dataset, coef_threshold, test_size, random_state):
    # split train test dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1], dataset.iloc[:,-1], test_size=test_size, random_state=random_state)
    
    # initiate and train a logreg model with balanced class weight
    clf = LogisticRegression(class_weight='balanced', random_state=random_state).fit(X_train, y_train)
    
    # use features with coefficient > coef_threshold
    df_features = pd.DataFrame()
    df_features['Feature'] = clf.feature_names_in_
    df_features['Coefficient'] = abs(clf.coef_[0])
    df_features = df_features.sort_values(by=['Coefficient'], ascending=False)
    df_features = df_features.reset_index()
    high_coef_features = df_features[abs(df_features['Coefficient']) > coef_threshold]['Feature']
    high_coef = df_features[abs(df_features['Coefficient']) > coef_threshold]['Coefficient']
    
    # retrain the model with only features with coefficient > coef_threshold
    clf_high_coef_features = LogisticRegression(class_weight='balanced', random_state=random_state).fit(X_train[high_coef_features], y_train)
    
    return clf_high_coef_features, high_coef_features, high_coef

