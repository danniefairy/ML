#autoencoder原資料壓縮成較小資料算是擷取出原資料精華(新的feature)的一種方式,如同PCA
#非監督學習
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
#不用Sequential
from keras.models import Model
from keras.layers import Dense,Input
%matplotlib notebook
import matplotlib.pyplot as plt

#取得資料
(x_train,_),(x_test,y_test)=mnist.load_data()

#資料預處理
#只用到input x_train(60000*28*28)
x_train=x_train.astype('float32')/255.0-0.5  #normalize -0.5~0.5
x_test=x_test.astype('float32')/255.0-0.5  #normalize -0.5~0.5
#x_train(60000*784)
x_train=x_train.reshape((x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))

#encoding dimension
#要把784壓縮成2
encoding_dim=2

#placeholder 佔位置
input_img=Input(shape=(784,))

#encoder layer
#input_img為輸入
#從784->128->64->10->2壓縮
encoded=Dense(128,activation="relu")(input_img)
encoded=Dense(64,activation="relu")(encoded)
encoded=Dense(10,activation="relu")(encoded)
encoded_output=Dense(encoding_dim,)(encoded)

#decoder layer
decoded=Dense(10,activation="relu")(encoded_output)
decoded=Dense(64,activation="relu")(decoded)
decoded=Dense(128,activation="relu")(decoded)
#因為輸入已經normalize成-0.5~0.5,所以用tanh(-1~1之間)效果好
decoded=Dense(784,activation="tanh")(decoded)

#建立autoencoder
autoencoder=Model(input=input_img,output=decoded)
#建立encoder
encoder=Model(input=input_img,output=encoded_output)
#compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')
#training
#autoencoder輸入以及label都一樣
autoencoder.fit(x_train,x_train,
               nb_epoch=20,
               batch_size=256,
               shuffle=True)

#plot
encoded_imgs=encoder.predict(x_test)
#兩個feature當作X,y軸
#c是顏色,所以用不同t_test當作不同顏色
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)
plt.show()