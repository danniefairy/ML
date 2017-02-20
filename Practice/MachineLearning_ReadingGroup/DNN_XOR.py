import theano
import theano.tensor as T
import random
import numpy as np
import time

#------------------------
'''
#variable can be scalar、matrix、tensor
a=theano.tensor.scalar()
b=theano.tensor.matrix()
c=theano.tensor.matrix("test matrix")
#a、b、c are symble without value
print(a,b,c)
'''
#------------------------
'''
#define function
x1=T.scalar()
x2=T.scalar()
x3=T.matrix()
x4=T.matrix()

y1=x1+x2
y2=x1*x2
#elementwise
y3=x3*x4
y4=T.dot(x3,x4)

#declare function
f=theano.function([x1,x2,x3,x4],[y1,y2,y3,y4])
print(f(3,5,[[1,2],[3,4]],[[5,6],[7,8]]))
'''
#------------------------
'''
#function example
x1=T.scalar()
x2=T.scalar()
y1=x1*x2
y2=x1**2+x2**0.5
#input value,output value
f=theano.function([x1,x2],[y1,y2])

z=f(2,4)
#theano output is numpy ndarray
print(z)
'''
#------------------------
'''
#gradient descent g=T.grad(y,x) => y must be scalar
x1=T.scalar('x1')
x2=T.scalar('x2')
y=x1*x2
#g=dy/dx1+dy/dx2
g=T.grad(y,[x1,x2])
f=theano.function([x1,x2],y)
f_grad=theano.function([x1,x2],g)
print(f(-2,2),f_grad(-2,2))
'''
'''
A=T.matrix()
B=T.matrix()

C=A*B
D=T.sum(C)
#D is a scalar but C isn't
g=T.grad(D,A)

y_grad=theano.function([A,B],[g,C,D])

X=[[1,2],[3,4]]
Y=[[5,6],[7,8]]

print(y_grad(X,Y))
'''
#------------------------
'''
#using share variables(just like global variable) in function needn't define as the input of the function
#using w.get_value() and w.set_value() function to change share variable
x=T.vector()
w=theano.shared(np.array([-1.,1.]))
b=theano.shared(0.)

z=T.dot(w,x)+b
y=1/(1+T.exp(-z))
neuron=theano.function(
                        inputs=[x],
                        outputs=y
                       )

y_hat=T.scalar()
cost=T.sum((y-y_hat)**2)

dw,db=T.grad(cost,[w,b])

gradient=theano.function(
                        inputs=[x,y_hat],
                        #outputs=[dw,db]
                        #using updates[(share variable,expression)] to  update share variable by expression 
                        updates=[(w,w-0.1*dw),(b,b-0.1*db)]
                         )

x=[1,-1]
y_hat=1
for i in range(100):
    #neuron(x) output y for gradient functino
    print(neuron(x))
    gradient(x,y_hat)
    #dw,db=gradient(x,y_hat)
    #w.set_value(w.get_value()-0.1*dw)
    #b.set_value(b.get_value()-0.1*db)
    print(w.get_value(),b.get_value())
'''
#------------------------
#Neural Network
#gpu only can work on float32

#XOR
import theano
import theano.tensor as T
import numpy as np

class Layer(object):
    def __init__(self,inputs,in_size,out_size,activation_function=None):
        self.W=theano.shared(np.random.randn(in_size,out_size))
        self.b=theano.shared(np.random.randn(out_size))
        self.z=T.dot(inputs,self.W)+self.b
        if activation_function is None:
            self.outputs=self.z
        else:
            self.outputs=activation_function(self.z)
            
#line up training sets and targets
x_data=np.array([[0,0],[0,1],[1,0],[1,1]])
y_data=np.array([[1],[0],[0],[1]])

x=T.dmatrix('x')
y=T.dmatrix('y')

#sigmoid 不好
l1=Layer(x,2,10,T.nnet.relu)
l2=Layer(l1.outputs,10,1,T.nnet.elu)
cost=T.mean((y-l2.outputs)**2)
dW1,db1,dW2,db2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])

learning_rate=0.05
train=theano.function(
                        inputs=[x,y],
                        outputs=cost,
                        updates=[(l1.W,l1.W-learning_rate*dW1),
                                 (l1.b,l1.b-learning_rate*db1),
                                 (l2.W,l2.W-learning_rate*dW2),
                                 (l2.b,l2.b-learning_rate*db2)]
                      )

prediction=theano.function(
                            inputs=[x],
                            outputs=l2.outputs
                           )

for i in range(1000):
    err=train(x_data,y_data)
    print(err)
print(prediction([[1,1],[0,0]]))