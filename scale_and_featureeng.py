import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression

def select_kbest(X, y, k):
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X, y)
    feature_mask = f_selector.get_support()

    f_feature = X.iloc[:,feature_mask].columns.tolist()
    return f_feature

def rfe(X,y,k):
    lm = LinearRegression()
    rfe = RFE(lm,k)
    rfe.fit(X, y)
    feature_mask_rfe = rfe.support_
    rfe_feature = X.iloc[:,feature_mask_rfe].columns.tolist()
    return rfe_feature

def scale_data(train,validate,test):
    '''Accepts train, validate, test data frames and applies min-max scaler
    return: train, validate, test scaled pandas dataframe'''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train_scaled, validate_scaled, test_scaled