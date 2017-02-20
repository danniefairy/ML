import numpy as np
import theano.tensor as T
import theano

#share value 是可以被改變的值 用在weights、bias
#dtype都要統一、初始下面設成0
state=theano.shared(np.array(0,dtype=np.float64),'state')

#要讓其他scalar、matrix和xxx用一樣的dtype用使用xxx.dtype
#'inc'可以不用
inc=T.scalar('inc',dtype=state.dtype)

#fuction內分別為輸入、輸出、update方式(update先中括號再小括號，逗點前方是原值，後方是用什麼值替代)
accumulator=theano.function([inc],state,updates=[(state,state+inc)])

#如果直接使用print(accumulator(10))，會先輸出原本state(0)然後才執行updates

#所以用xxx.get_value()
accumulator(15)
print(state.get_value())

#更改內容用xxx.set_value(input here)
state.set_value(-50)
print(state.get_value())

#暫時取得替代的值，state在這裡就暫時變成5帶入
temp=1/(T.exp(-state)+inc)
a=T.scalar(dtype=state.dtype)
#givens是指想把什麼代替成什麼
temp_share=theano.function([inc,a],temp,givens=[(state,a)])
print(temp_share(10,5))