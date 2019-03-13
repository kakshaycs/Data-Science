# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:26:19 2018

@author: kaksh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

    
def scallingData(X):
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    return X

    
def missingdataString(X):
    df=pd.DataFrame(X)
    df=df.fillna(method = "ffill")
    return df.iloc[:,:].values
        
        
    
    
def categoriseData(Z,X,arr):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    onehotencoder = OneHotEncoder(categorical_features = [0])
    for i in arr:
       xx=X[:,[i]]
       xx[:, 0] = labelencoder_X.fit_transform(xx[:, 0])
       xx = onehotencoder.fit_transform(xx).toarray()
       xx=xx[:,1:]
       Z=np.append(xx,Z,axis=1)
    return Z


def missingData(X):
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, :])
    X[:, :] = imputer.transform(X[:, :])
    return X
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    