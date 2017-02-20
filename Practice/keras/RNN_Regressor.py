import numpy as np
np.random.seed(1337)
%matplotlib notebook
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,TimeDistributed,Dense
from keras.optimizers import Adam

BATCH_START=0
#取20步
TIME_STEPS=20
#每一個批次大小
BATCH_SIZE=50
#一個時間點對應到一個數據
INPUT_SIZE=1
#輸出一個時間點也是一個數據
OUTPUT_SIZE=1
#每一個RNN內NN內的node數目
CELL_SIZE=20
LR=0.001

def get_batch():
    global BATCH_START,TIME_STEPS
    #從BATCH_START~BATCH_SIZE+TIME_STEPS*BATCH_SIZE
    #而BATCH_START會每一次增加TIME_STEPS,而把這些變成BATCH_SIZE*TIME_STEPS大小的矩陣,後面除上10*np.pi是為了normalize
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))/(10*np.pi)
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START+=TIME_STEPS
    #把seq,res[[20個...],[20個...]...50個]改成[[[1個],[1個],[1個]...共20個1個..[1個]]...50個](20為TIME_STEPS)(50為BATCH_SIZE)
    return[seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]


#建立model
model=Sequential()

#建立LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
    output_dim=CELL_SIZE,
    #每一個時間點都會輸出output
    #default=false,在最後一個TIME_STEPS才會輸出
    return_sequences=True,
    #Batch之間有聯繫
    #每一個圖片行28個pixel會輸出一個state來描述這行,stateful=True表示這個state(狀態)會有聯繫的傳下去
    stateful=True,
    ))
#output layer
#每一個TIME_STEPS會有一個OUTPUT,所以DENSE對所有做全連接計算
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

#optimizer
adam=Adam(LR)
model.compile(
    optimizer=adam,
    loss='mse',
)

print("Training~~~~~")
for step in range(501):
    X_batch,Y_batch,xs=get_batch()
    cost=model.train_on_batch(X_batch,Y_batch)
    pred=model.predict(X_batch,BATCH_SIZE)
    #畫
    plt.plot(xs[0,:],Y_batch[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')
    plt.ylim((-1.2,1.2))
    plt.draw()
    try:
        plt.pause(0.5)
    except Exception:
        pass
    if step%10==0:
        print('train cost: ',cost)