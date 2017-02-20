from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

#係數
print(model.coef_)
#常數(和y軸交點)
print(model.intercept_)

#model評分,先放要訓練的,再放相對應的label(model為LinearRegression時使用R^2)
print(model.score(data_X,data_y))

#看一下預測前後結果
print(model.predict(data_X)[:10])
print(data_y[:10])