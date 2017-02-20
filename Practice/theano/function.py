import numpy as np
import theano.tensor as T
from theano import function
#呼叫theano用
import theano

#activation function
x=T.dmatrix('x')
s=1/(1+T.exp(-x))
logistic=function([x],s)
#記得要多加個括號
print(logistic([[1,2],[5,6]]))

#multiply output for a function
a,b=T.dmatrices('a','b')
diff=a-b
abs_diff=abs(diff)
#多個輸入、輸出要用中刮號表示
f=function([a,b],[diff,abs_diff])
#自己打矩陣記得要多加個括號(小括號、中括號)
#print(f([[1,2]],[[3,4]]))
x1,x2=f(np.ones((2,2)),np.arange(4).reshape((2,2)))
print(x1,x2)

#name for a function
x,y,w=T.dscalars('x','y','w')
z=(x+y)*w
#設定默認值
f=function([x,theano.In(y,value=1),theano.In(w,value=2,name='weights')],z)
print(f(20,5,weights=4))