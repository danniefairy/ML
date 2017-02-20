import theano
from sklearn.datasets import load_boston
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook

#亂數
np.random.seed(100)

class Layer(object):
    def __init__(self,inputs,in_size,out_size,activation_function=None):
        self.W=theano.shared(np.random.normal(0,1,(in_size,out_size)))
        self.b=theano.shared(np.zeros((out_size,))+0.01)
        self.Wx_plus_b=T.dot(inputs,self.W)+self.b
        self.activation_function=activation_function
        if activation_function is None:
            self.outputs=self.Wx_plus_b
        else:
            self.outputs=activation_function(self.Wx_plus_b)

#標準化
def normalization(data):
    xn_max=np.max(data,axis=0)
    xn_min=np.min(data,axis=0)
    xn=(data-xn_min)/(xn_max-xn_min)
    return xn
                             
#讀取資料
x_data=load_boston().data
#標準化
x_data=normalization(x_data)
y_data=load_boston().target[:,np.newaxis]
                             
#cross validation
x_train,y_train=x_data[:400],y_data[:400]
x_test,y_test=x_data[400:],y_data[400:]
                             
x=T.dmatrix("x")
y=T.dmatrix("y")
                             
#定義類神經網路層,因為共有13個features所以input_size=13
l1=Layer(x,13,50,T.tanh)
l2=Layer(l1.outputs,50,1,None)

#cost函數
#基本
#cost=T.mean(T.square(y-l2.outputs))
#regularization l2
cost=T.mean(T.square(y-l2.outputs))+0.1*((l1.W**2).sum()+(l2.W**2).sum())
#regularization l1
#cost=cost+0.1*(abs((l1.W).sum())+abs((l2.W).sum()))
gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])

#training
learning_rate=0.01
train=theano.function(
    inputs=[x,y],
    updates=([l1.W,l1.W-learning_rate*gW1],
            [l1.b,l1.b-learning_rate*gb1],
            [l2.W,l2.W-learning_rate*gW2],
            [l2.b,l2.b-learning_rate*gb2])
)

#計算cost
compute_cost=theano.function(inputs=[x,y],outputs=cost)

#紀錄cost
train_err_list=[]
test_err_list=[]
learning_time=[]
for i in range(1000):
    train(x_train,y_train)
    if i%10==0:
        #training data 
        train_err_list.append(compute_cost(x_train,y_train))
        #test data
        test_err_list.append(compute_cost(x_test,y_test))
        #time
        learning_time.append(i)

#圖像化        
plt.plot(learning_time,train_err_list,"r-")
plt.plot(learning_time,test_err_list,"b--")
plt.show()