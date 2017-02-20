#用learning curve 可視畫學習過程
from sklearn.learning_curve import learning_curve
#輸入digit data
from sklearn.datasets import load_digits
#model
from sklearn.svm import SVC
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

digits=load_digits()
X=digits.data
y=digits.target

#輸出各種對應training size 的結果,依序為 model、input data、input label、選擇分成幾分、計分方式、各種training size 選擇
#這裡的範例是若gamma=0.01則就會overfitting
train_sizes,train_loss,test_loss=learning_curve(
        SVC(gamma=0.001),X,y,cv=10,scoring='mean_squared_error',
        train_sizes=[0.1,0.25,0.5,0.75,1]
)
#train_loss、test_loss各有10個(因為cv=10),取平均
#loss取負數是因為loss輸出負值
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='Training')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='Cross-validation')

plt.xlabel('sample')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()























