from sklearn import datasets
#import 一個ML model
from sklearn.linear_model import LinearRegression
%matplotlib notebook
import matplotlib.pyplot as plt

loaded_data=datasets.load_boston()
#取得X(features)
data_X=loaded_data.data
#取得y(label)
data_y=loaded_data.target

#定義model,下面為使用默認值
model=LinearRegression()
#訓練
model.fit(data_X,data_y)

#使用預測
print(model.predict(data_X[:10,:]))
print(data_y[:10])


'''
流程:
    數據預處理: normalize
    選擇model、選擇參數
'''

#自創數據點,noise可以增加離散度
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
#圖像化
plt.scatter(X,y)
plt.show()