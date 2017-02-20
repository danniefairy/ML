import numpy as np
np.random.seed(200)
from keras.models import Sequential
from keras.layers import Dense
%matplotlib notebook
import matplotlib.pyplot as plt
#區分train、test data
from sklearn.cross_validation import train_test_split

#建立資料
X=np.linspace(-1,1,200)
#shuffle資料
np.random.shuffle(X)
#增加noise,weight=0.5、bias=2是我要學出來的參數
Y=0.5*X+2+np.random.normal(0,0.05,(200,))

#印出來看看
#plt.scatter(X,Y)
#plt.show()

#X、Y分成160、40筆train、test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#建立從input到output間所有的layers(從第一層開始定義)
model=Sequential()
#增加一層，注意第一層input_dim要跟資料features相符
model.add(Dense(output_dim=1,input_dim=1))
#第二層之後input_dim就會自動默認
#model.add(Dense(output_dim=1))

#選擇loss function以及最佳化(compile)
#loss為mean square error,optmizer為stochastic gradient descent
model.compile(loss='mse',optimizer='sgd')

#train
print('Training!')
for step in range(301):
    #依據每一批每一批的數據訓練
    cost=model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print('train cost:',cost)
        
#test
print('\nTest!')
#batch_size=40是因為test data有40個
cost=model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
#取得weight、bias
W,b=model.layers[0].get_weights()
print('Weights:',W)
print('Bias:',b)

#比較原本真實test data跟預測test data差異
Y_pred=model.predict(X_test)
plt.scatter(X_test,Y_test,color='g')
plt.plot(X_test,Y_pred,color='r')
plt.show()