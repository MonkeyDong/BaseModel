from sklearn import datasets
import numpy as np

iris=datasets.load_iris()

iris_x=iris.data
iris_y=iris.target

indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris_x_train, iris_y_train)
iris_y_predict = knn.predict(iris_x_test)
