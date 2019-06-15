from sklearn.ensembleimport GradientBoostingClassifier
from sklearn.datasets import load_iris

iris = load_iris()
gbm0= GradientBoostingClassifier(random_state=10)
gbm0.fit(iris.data, iris.target)
y_pred= gbm0.predict(iris.data)
