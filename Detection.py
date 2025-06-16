import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('PCOS Dataset.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#print(dataset.isnull().sum()) no missing values

#period flow is a categorical variable so we do one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[15])],remainder='passthrough') #passthrough because we want the age and salary columns to be as it is present in the dataset
X=np.array(ct.fit_transform(X))
print(X.shape)

#splitting into train and test datasets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


#feature scaling 
#print(X_test[:,0])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scale_indexes = [1, 2, 3, 4, 5, 15, 16, 17, 31]
X_train[:,scale_indexes]=sc.fit_transform(X_train[:,scale_indexes]) #3 and 4 become our new age AND salary after we do onehot encoding of data
X_test[:,scale_indexes]=sc.transform(X_test[:,scale_indexes])
print (X_test)