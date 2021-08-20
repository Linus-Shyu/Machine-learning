#画个logo，哈哈


print("----------------------------------------------------------------")
print("X          X     XXXXX        X   X       X     X         X   X")
print("X          X     X   X        X   X          X            X   X")
print("X          X     X   X        X   X         X  X          X   X")
print("X X X X    X     X   X        XXXXX      X       X        XXXXX")
print("----------------------------------------------------------------")


#导入库，导入算法


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#获取小规模鸢尾花数据集，并将它存放到(iris)这个变量之中


iris=datasets.load_iris()


#获取鸢尾花所有的数据，并将它存放到(iris_X)这个变量之中


iris_X=iris.data


#获取目标值，并将它存放到(iris_Y)这个变量之中


iris_Y=iris.target


#将数据进行划分，一部分用于测试一部分用于训练


X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size=0.3)


#运用K近邻算法，并将它存放到(KNN)这个变量之中


KNN=KNeighborsClassifier()


#开始训练数据


KNN.fit(X_train,Y_train)


#使用训练数据预测结果,并将它打印出来


print(KNN.predict(X_test))


#将两组数据分割，比较好看一些
print("---------------------------------------------------------------------------")


#将测试数据打印出来，进行比对


print(Y_test)













