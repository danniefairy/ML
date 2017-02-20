import numpy as np
import theano.tensor as T
import theano
#保存
import pickle

#隨機設定
np.random.seed(100)

#計算出正確比例
def compute_accuracy(y_target,y_predict):
    correct_prediction=np.equal(y_predict,y_target)
    accuracy=float(np.sum(correct_prediction))/float(len(correct_prediction))
    return accuracy

rng=np.random

#樣本大小
N=400
#特徵數
features=784

#建立dataset=(input_value,target_class),下列例子分成兩類
#randint low=0 high=2 表示 輸出範圍界在0~2之間(有包含上下限)
D = (rng.randn(N, features), rng.randint(size=N, low=0, high=2))

#定義input
x=T.dmatrix('x')
y=T.dvector("y")

#初始化weight、bias
w=theano.shared(rng.randn(features),name='w')
b=theano.shared(0.01,name='b')

#classification 前置作業
#機率
p_1=T.nnet.sigmoid(T.dot(x,w)+b)
prediction=p_1>0.5

#有model後這些就不需要了------------------------------
#或是用 crossentropy=T.nnet.binary_crossentropy(p_1, y)
crossentropy=-y*T.log(p_1)-(1-y)*T.log(1-p_1)
#0.01*(w**2).sum() 防止overfitting
cost=crossentropy.mean()+0.01*(w**2).sum()
gW,gb=T.grad(cost,[w,b])



#train
learning_rate=0.05
train=theano.function(
    inputs=[x,y],
    outputs=[prediction,crossentropy.mean()],
    updates=((w,w-learning_rate*gW),
            (b,b-learning_rate*gb))
)
#有model後這些就不需要了------------------------------

#預測
predict=theano.function(inputs=[x],outputs=prediction)

#有model後這些就不需要了------------------------------
#training
for i in range(5000):
    #input_value,target_class
    pred,err=train(D[0],D[1])
    if i%50==0:
        #print('costs:',err)
        print('accuracy:',compute_accuracy(D[1],predict(D[0])))
        
        
#原始資料V.S.預測結果
print('target value for D:')
print(D[1])
print('prediction on D:')
print(predict(D[0]))

#保存model
with open('save/model.pickle','wb') as file:
    #把讀出過的weight、bias讀出並且存取
    model=[w.get_value(),b.get_value()]
    pickle.dump(model,file)
    print(model[0][:10])
    print("accuracy:",compute_accuracy(D[1],predict(D[0])))    
#有model後這些就不需要了------------------------------

'''
#讀取model
with open('save/model.pickle','rb') as file:
    model=pickle.load(file)
    w.set_value(model[0])
    b.set_value(model[1])
    print(w.get_value()[:10])
    print("accuracy:",compute_accuracy(D[1],predict(D[0])))
'''