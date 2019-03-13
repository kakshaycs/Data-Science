# machine learning kaggle titanic


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learn as LL

# Importing the dataset
dataset = pd.read_csv('train.csv')

# check how many column have null values and type of columns
print(dataset.isnull().sum())
dataset.dtypes


X=dataset.iloc[:,[5,6,7,9]].values
Xc=dataset.iloc[:,[2,4,11]].values
y=dataset.iloc[:,[1]].values

X=LL.missingData(X)
X=LL.scallingData(X)
Xc=LL.missingdataString(Xc)
X=LL.categoriseData(X,Xc,[0,1,2])



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)






# fitting Naive bayes to the training set

from sklearn.naive_bayes import GaussianNB
classifierNaive= GaussianNB()
classifierNaive.fit(X_train,y_train)


# 78.12 %

# fitting support vector machine to the training set

from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'rbf', random_state=0)
classifierSVM.fit(X_train,y_train)






#  predicting test result
y_pred=classifierSVM.predict(X_test)




# MAking the confusion MAtrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




# finding result

dataset1 = pd.read_csv('test.csv')
X_Test=dataset1.iloc[:,[4,5,6,8]].values
Xc_Test=dataset1.iloc[:,[1,3,10]].values


X_Test=LL.missingData(X_Test)
X_Test=LL.scallingData(X_Test)
Xc_Test=LL.missingdataString(Xc_Test)
X_Test=LL.categoriseData(X_Test,Xc_Test,[0,1,2])


#  predicting test result
result=classifierSVM.predict(X_Test)


dataset2 = pd.read_csv('gender_submission.csv')
result2 = dataset2.iloc[:,[1]].values


r1 = dataset2.iloc[:,[0]].values
result=Z=np.append(r1,result,axis=1)


submission = pd.DataFrame({
        "PassengerId":dataset2["PassengerId"],
        "Survived":result
        })

submission.to_csv('submission.csv',index=False)









