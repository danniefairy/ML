from sklearn import preprocessing
import numpy as np
#分開training data、test data
from sklearn.cross_validation import train_test_split
#生成classification數據
from sklearn.datasets.samples_generator import make_classification
#處理model
from sklearn.svm import SVC
%matplotlib notebook
import matplotlib.pyplot as plt

'''
a=np.array([[10,2.3,-5],
            [200,400,800],
           [120,40,60]],dtype=np.float64)

print(a)
#normalization後結果
print(preprocessing.scale(a))
'''
#生成data 
X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,random_state=22,n_clusters_per_class=1,scale=100)

#看數據結構
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()

#可以決定normalization範圍 (preprocessing.minmax_scale(X,feature_range=(-1,1)))
X=preprocessing.scale(X)

#分開training data、test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#model
clf=SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))