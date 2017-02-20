import numpy as np
import theano.tensor as T
from theano import function
'''
#宣告
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y
f=function([x,y],z)
print(f(2,3))

#查看函數組成
from theano import pp
print(pp(z))
'''
#矩陣
X=T.dmatrix('X')
Y=T.dmatrix('Y')
Z=T.dot(X,Y)
F=function([X,Y],Z)
print(F(np.ones((2,3)),2*np.ones((3,1))))