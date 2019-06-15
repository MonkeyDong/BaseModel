from sklearn.model_selection import cross_val_score 
from sklearn.datasets import make_blobs 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0) 
scores1 = cross_val_score(clf1, X, y) 
print(scores1.mean())

clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores2 = cross_val_score(clf2, X, y)
print(scores2.mean())

clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores3 = cross_val_score(clf3, X, y)
print(scores3.mean())
