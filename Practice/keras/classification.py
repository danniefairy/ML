import numpy as np
np.random.seed(200)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
#optimizer 可以加快運算速度,例如用stochastic gradient descent 以批次的方式將資料餵入NN,而不是一整個餵進去
#RMSprop=AdaGrad(增加sgd搖擺的阻力,使搖擺幅度降低)+Momentum(強迫訓練時跟著慣性走)
from keras.optimizers import RMSprop

#讀取資料:手寫資料
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#X_train.shape[0]代表找出X_train的大小(多少字)
#一開始X_train[i]是由[28][28]組成,下面步驟是把X_train強制設為只有60,000個維度
#.shape[0]代表算出X_train的每第一筆的資料數(60,000)
#reshape(,-1)代表變成一個row而已
#把原本X_train=60000*28*28變成X_train=60000*784
X_train=X_train.reshape(X_train.shape[0],-1)/255.0
X_test=X_test.reshape(X_test.shape[0],-1)/255.0
#把原本y_train=5變成y_train=[0,0,0,0,0,1,0,0,0,0] (One Hot Encoding)
y_train=np_utils.to_categorical(y_train,nb_classes=10)
y_test=np_utils.to_categorical(y_test,nb_classes=10)

#建立model
model=Sequential([
        Dense(32,input_dim=28*28),
        Activation('relu'),
        #不用再定義input_dim
        Dense(10),
        Activation('softmax')
    ])

#最佳化
rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-8)

#加入矩陣
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    #計算過程中可以計算出來的矩陣
    metrics=['accuracy']
)

print('training!')
#training
#nb_epoch代表訓練多少次
model.fit(X_train,y_train,nb_epoch=2,batch_size=32)

print('testing!')
#evaluate
loss,accuracy=model.evaluate(X_test,y_test)

print('loss:',loss)
print('accuracy:',accuracy)