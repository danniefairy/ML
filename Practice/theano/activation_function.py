#透過激活函數可以強化輸出的結果
#中間層(hidden layer): relu、tanh、softplus......
#輸出層(output layer): sigmoid、softmax
#回歸(regression)問題可以用線性輸出(linear output layer)

import numpy as np
import theano.tensor as T
import theano
#需要多加下面這一行在jupyter notebooks讓圖顯示
#%matplotlib notebook #在jupyter上才要
import matplotlib.pyplot as plt
import time
'''
定義layer
l1=Layer(輸入資料,輸入資料大小,輸出資料大小,activation_function)
l2=Layer(從l1輸出資料,從l1輸出大小,輸出資料大小,activation_function)
'''

class Layer(object):
    def __init__(self,inputs,in_size,out_size,activation_function=None):
        #建立W(weights)為一個in_size raw、out_size column 的矩陣，其值為0~1標準分布
        self.W=theano.shared(np.random.normal(0,1,(in_size,out_size)))
        #建立b(bias)為一個out_size 大小的矩陣
        self.b=theano.shared(np.zeros((out_size))+0.1)
        #Wx+b
        self.Wx_plus_b=T.dot(inputs,self.W)+self.b
        #activation function
        self.activation_function=activation_function
        if activation_function is None:
            self.outputs=self.Wx_plus_b
        else:
            self.outputs=activation_function(self.Wx_plus_b)
            

#假數據
#把一維矩陣變成多維
#dtype=float64
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

#顯示資料
#plt.scatter(x_data,y_data)
#plt.show()

#定義input,d為float64
x=T.dmatrix('x')
y=T.dmatrix('y')

#增加layer,in_size=1是因為x只有一個屬性,out_size=10自己定義的
l1=Layer(x,1,10,T.nnet.relu)
#output layer 的大小out_size=1是因為y也是只有一個維度
l2=Layer(l1.outputs,10,1,None)

#計算cost(平均誤差)
cost=T.mean(T.square(l2.outputs-y))

#Gradient 計算(每次weight、bias變化量),後面放參數
gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])

#Gradient Descent 應用
learning_rate=0.05
#input x是因為呼叫cost函數會用到x、y來計算出l2.outputs
train=theano.function(
    inputs=[x,y],
    outputs=cost,
    #每一個要更新的東西用小括號代替
    updates=[(l1.W,l1.W-learning_rate*gW1),
             (l1.b,l1.b-learning_rate*gb1),
             (l2.W,l2.W-learning_rate*gW2),
             (l2.b,l2.b-learning_rate*gb2)]
    )

#預測,只需要input
predict=theano.function(inputs=[x],outputs=l2.outputs)


#印出input data
fig=plt.figure()
#第一行、第一列、第一個
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#show(block=True)為默認值
plt.ion()
plt.show()


for i in range(1000):
    #training
    err=train(x_data,y_data)
    if i%50==0:
        #print(err)
        #圖像化我每一步進步的結果
        #避免每一條線重疊出現,所以移除上一個迴圈新生成的線
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predicted_value=predict(x_data)
        lines=ax.plot(x_data,predicted_value,'r-',lw=5)
        try:
            plt.pause(1)
        except Exception:
            pass
