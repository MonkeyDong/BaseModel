from sklearn.svm import SVC
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train = #(n,[m])
y_train = #[0,1,0,1...]
X_test = 
y_test = 

clf = SVC()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
