#导入numpy，导入knn算法模型，导入数据集。
import numpy as np
from numpy.core.arrayprint import set_string_function
from numpy.lib.index_tricks import index_exp
from sklearn.metrics.pairwise import distance_metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets


#获取鸢尾花数据集。
datasets.load_iris()


#创建变量(iris)存放获取到的鸢尾花数据集。
iris=datasets.load_iris()


#设置变量(X,Y)用于存放data和target。
X=iris['data']
Y=iris['target']


#将数据分成两部分，一部分用于训练，一部分用于测试。
#将数据顺序打乱，测试，训练效果会更好。
index=np.arange(150)
np.random.shuffle(index)


X_train,X_test=X[index[:100]],X[index[100:]]
Y_train,Y_test=Y[index[:100]],Y[index[-50:]]


#声明算法给它五个同类。
KNN=KNeighborsClassifier(n_neighbors=5)


#对数据进行预测。
KNN.fit(X_train,Y_train)
N=KNN.predict(X_test)


#预测结果和真实结果对比。
print(N)
print('-------------------------------------------------')
print(Y_test)


#准确率计算。
A=KNN.score(X_test,Y_test)
print(A)
