import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('PCOS Dataset.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#print(X.shape)

#print(dataset.isnull().sum()) no missing values

#period flow is a categorical variable so we do one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[15])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
#print(X.shape)

#splitting into train and test datasets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


#feature scaling 
#print(X_test[:,0])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scale_indexes = [2, 3, 4, 6, 31]
X_train[:,scale_indexes]=sc.fit_transform(X_train[:,scale_indexes]) 
X_test[:,scale_indexes]=sc.transform(X_test[:,scale_indexes])
#print (X_test)

#classification models
#knn
from sklearn.neighbors import KNeighborsClassifier
classifier1=KNeighborsClassifier(n_neighbors=5)
classifier1.fit(X_train,y_train)
y_pred1=classifier1.predict(X_test)
y_pred1 = y_pred1.reshape(len(y_pred1), 1)
y_test = y_test.reshape(len(y_test), 1)
#print(np.concatenate((y_pred1, y_test), axis=1))

from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,y_pred1))
cm=confusion_matrix(y_test,y_pred1)
print(cm)

#kernel svm
from sklearn.svm import SVC
classifier2=SVC(random_state=0,kernel='rbf')
classifier2.fit(X_train,y_train)
y_pred2=classifier2.predict(X_test)
y_pred2 = y_pred2.reshape(len(y_pred2), 1)
y_test = y_test.reshape(len(y_test), 1)
#print(np.concatenate((y_pred2, y_test), axis=1))

from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,y_pred2))
cm=confusion_matrix(y_test,y_pred2)
print(cm)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier3=GaussianNB()
classifier3.fit(X_train,y_train)

y_pred3 = classifier3.predict(X_test)
print(accuracy_score(y_test,y_pred3))
cm=confusion_matrix(y_test,y_pred3)
print(cm)

#Decision tree
from sklearn.tree import DecisionTreeClassifier
classifier4=DecisionTreeClassifier(criterion='entropy')
classifier4.fit(X_train,y_train)

y_pred4=classifier4.predict(X_test)
print(accuracy_score(y_test,y_pred4))
cm=confusion_matrix(y_test,y_pred4)
print(cm)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier5=RandomForestClassifier(n_estimators=150,criterion='entropy')#default n=100
classifier5.fit(X_train,y_train)
y_pred5=classifier5.predict(X_test)

print(accuracy_score(y_test,y_pred5))
cm=confusion_matrix(y_test,y_pred5)
print(cm)
