from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

digits=load_digits()
X=digits.data
y=digits.target
print(X)
'''
#這個方法可以讓gamma是有range的,無training size,而是用param_range(gamma的改變)代替x軸
#從-6~-2.3取5個數字
param_range=np.logspace(-6,-2.3,5)
train_loss,test_loss=validation_curve(
    SVC(),X,y,cv=10,param_name='gamma',param_range=param_range,scoring='mean_squared_error'
)

#axis=1是因為每一個row的元素來平均
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color='r',label='training')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='cross validation')

plt.xlabel('gamma')
plt.ylabel('loss')
plt.show()
'''