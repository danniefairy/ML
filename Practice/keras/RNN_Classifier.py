import numpy as np
#隨機位置
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
#RNN序列數據上的神經網路
from keras.layers import SimpleRNN,Activation,Dense
from keras.optimizers import Adam

#把影像變成一行一行的序列
#總共有高28個PIXEL ARRAY的資料
TIME_STEP=28
#每一個PIXEL ARRAY有28個元素
INPUT_SIZE=28
#每一批生成多少圖片
BATCH_SIZE=50
#每一批資料起始位置，初始為0
BATCH_INDEX=0
#輸出值,因為這裡我們共有十個輸出[0,0...1,0,..0]
OUTPUT_SIZE=10
#RNN HIDDEN UNITS(每一個RNN內單一NN的SIZE)
CELL_SIZE=10
#LEARNING RATE
LR=0.001

#DATA
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#一開始X_train[i]是由[28][28]組成,下面步驟是把X_train強制設為只有60,000個維度
#.shape[0]代表算出X_train的每第一筆的資料數(60,000)
#reshape(,-1)代表變成一個row而已
#把原本X_train=60000*28*28變成X_train=60000*784
X_train=X_train.reshape(-1,28,28)/255.0  #normalize 255.0變成float
X_test=X_test.reshape(-1,28,28)/255.0    #normalize 255.0變成float
#把原本y_train=5變成y_train=[0,0,0,0,0,1,0,0,0,0] (One Hot Encoding)
y_train=np_utils.to_categorical(y_train,nb_classes=10)
y_test=np_utils.to_categorical(y_test,nb_classes=10)

#建立RNN model
model=Sequential()

#RNN cell
#input layer
#SimpleRNN activation default='tanh'
model.add(SimpleRNN(
    batch_input_shape=(BATCH_SIZE,TIME_STEP,INPUT_SIZE),
    output_dim=CELL_SIZE,
    ))
#output layer
#SimpleRNN activation default='tanh'
model.add(Dense(OUTPUT_SIZE))
#SimpleRNN activation default='tanh'
model.add(Activation('softmax'))

#optimizer
adam=Adam(LR)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#training
for step in range(4001):
    #一批一批資料累加,:=28 TIME_STEP,:=28 INPUT_SIZE=>3維數據
    X_batch=X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:,:]
    #2維數據
    Y_batch=y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:]
    #一個一個batch加進來
    cost=model.train_on_batch(X_batch,Y_batch)
    
    #batch更新
    BATCH_INDEX+=BATCH_SIZE
    #如果BATCH_INDEX>所有SIZE則歸零重新取BATCH
    BATCH_INDEX=0 if BATCH_INDEX>=X_train.shape[0] else BATCH_INDEX
    
    if step%500==0:
        cost,accuracy=model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
        print('test cost: ',cost,'test accuracy: ',accuracy)