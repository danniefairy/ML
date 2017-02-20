import numpy as np
#穩定compile,在Secuqential以前
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
#Adam是由Momentum(增加直線衝勁)+AdaGrad(讓下降搖擺緩慢)
from keras.optimizers import Adam
import time

#CNN基本結構
#(CONV、RELU、POOL)+....+FC

#--------------------------------------------

#數據讀取及預處理
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#-1代表這個list個數,1代表channel個數(此區因為黑白所以只需要一個),把每一個28*28拉成1*784
X_train=X_train.reshape(-1,1,28,28)
X_test=X_test.reshape(-1,1,28,28)
#把label=5改成[0,0,0,0,0,1,0,0,0,0]
y_train=np_utils.to_categorical(y_train,nb_classes=10)
y_test=np_utils.to_categorical(y_test,nb_classes=10)

#建立model
model=Sequential()

#--------------------------------------------
#Convolution layer1
#輸出大小(32,28,28)
model.add(Convolution2D(
    #用32個濾波器掃過一張圖片,每一個濾波器會產生一個特徵圖片
    #所以32個濾波器就會產生32層高度的圖片
    #所以變成28*28*32的結構
    nb_filter=32,
    #濾波器大小
    nb_row=5,
    nb_col=5,
    #padding:由於經過convolution可能會使輸出圖像尺寸減小,所以一開始在圖像邊緣補0就可以先增大輸入,使輸出一致
    border_mode='same',
    #1個高度(channel),28個寬度,28個長度
    input_shape=(1,28,28),
    #若用tensorflow要加這一行
    dim_ordering='th', 
    ))
#convolution後使用activation function(relu)
model.add(Activation('relu'))

#pooling layer1:對從convolution,activation function後的資料做長跟寬的減少(不減少深度)
#輸出大小(32,28/2,28/2)=(32,14,14)
model.add(MaxPooling2D(
    #每2高度2寬度取一個值,所以28*28->14*14
    pool_size=(2,2),
    #跳多少圖片pixels(長寬各跳兩個)
    strides=(2,2),
    #padding
    border_mode='same',
    ))

#--------------------------------------------

#Convolution layer2:
#輸出大小(64,14,14)
#依序為:filter個數、filter長寬、padding
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))

#pooling layer2:輸出大小(64,14/2,14/2)=(64,7,7)
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

#--------------------------------------------

#Fully Connected layer1
#把三維(64,7,7)輸入抹平成一維
model.add(Flatten())
#1024 nodes
model.add(Dense(1024))
model.add(Activation('relu'))

#Fully Connected layer2(output layer)
#輸出有10個分類
model.add(Dense(10))
model.add(Activation('softmax'))

#--------------------------------------------

#optimizer定義
adam=Adam(lr=1e-4)

#compile model
model.compile(optimizer=adam,
            loss='categorical_crossentropy',
             metrics=['accuracy'])

#--------------------------------------------

print('Training!')
#epoch總樣本訓練循環次數、batch一層逐一樣本訓練分成多少個,verbos取消進度條但可以在keras少bug
model.fit(X_train,y_train,nb_epoch=1,batch_size=32,verbose=0)
#加delay for bug
time.sleep(0.1)

print('Testing!')
loss,accuracy=model.evaluate(X_test,y_test)

print('\ntest loss: ',loss)
print('\ntest accuracy: ',accuracy)