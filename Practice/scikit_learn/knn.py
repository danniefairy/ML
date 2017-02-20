import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#KNN
from sklearn.neighbors import KNeighborsClassifier

#讀取資料
iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
'''
x為特徵,y為label
print(iris_X)
print(iris_y)
'''

#分割資料,train70% test30%,且會打亂數據
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)

#定義使用的函數
knn=KNeighborsClassifier()
#訓練
knn.fit(X_train,y_train)
'''
使用predict 把你要的東西帶入訓練好的函數
print(knn.predict(X_test))
print(y_test)
'''
