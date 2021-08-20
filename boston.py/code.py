#导入库，导入算法,导入可视化工具
print("----------------------------------------------------------------")
print("X          X     XXXXX        X   X       X     X         X   X")
print("X          X     X   X        X   X          X            X   X")
print("X          X     X   X        X   X         X  X          X   X")
print("X X X X    X     X   X        XXXXX      X       X        XXXXX")
print("----------------------------------------------------------------")

from typing import no_type_check
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt



#小规模获取波士顿房价数据集并将它存放到(loaded_data)这个变量之中


loaded_data=datasets.load_boston()


#将波士顿房价的数据存放到(data_X)这个变量中


data_X=loaded_data.data


#获取波士顿房价的目标值


data_Y=loaded_data.target


#定义模型


model=LinearRegression()


#让模型进行训练


model.fit(data_X,data_Y)


#用训练值对比真实值


print(model.predict(data_X[:4:]))
print("--------------------------------------------------")
print(data_Y[:4])



#创造数据点，并存放到(X,Y)这两个变量之中


X,Y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)


#将数据进行图像化


plt.scatter(X,Y)
plt.show()


#对模型进行打分


print(model.score(data_X,data_Y))


 





