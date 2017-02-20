from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
X=iris.data
y=iris.target

'''
#只會取一種test

#random_state一樣會有一樣的隨機結果
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)

#model,找出數據點附近的5個neighbor的值並中和
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
#預測
y_pred=knn.predict(X_test)
#算分
print(knn.score(X_test,y_test))
'''

'''
#一種neighbor數的多個cross_val_score

from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
#X,y會各自分成很多組(cv=5組)
scores=cross_val_score(knn,X,y,cv=5,scoring='accuracy')
#各組
print(scores)
#平均
print(scores.mean())
'''

#多種neighbors比較

%matplotlib notebook
import matplotlib.pyplot as plt

score_list=[]
k_range= range(1,30)
from sklearn.cross_validation import cross_val_score
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    #classification,scores越大越好
    scores=cross_val_score(knn,X,y,cv=5,scoring='accuracy')
    #regression,loss越小越好
    #loss=-cross_val_score(knn,X,y,cv=5,scoring='mean_squared_error')
    score_list.append(scores.mean())
    #score_list.append(loss.mean())
plt.plot(k_range,score_list)
plt.xlabel("k neighbors")
plt.ylabel("mean value")
plt.show()