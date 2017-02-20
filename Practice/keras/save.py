import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#假資料
X=np.linspace(-1,1,200)
#打亂資料
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200))
X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]
model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))
model.compile(loss='mse',optimizer='sgd')
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train)
    
#save
print('before save: ',model.predict(X_test[0:2]))
#HDF5 ,pip installh5py
model.save("keras_save.h5")
#刪除model
del model

#load
model=load_model("keras_save.h5")
print('after load: ',model.predict(X_test[0:2]))