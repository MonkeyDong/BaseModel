import numpy as np
from sklearn.decomposition import PCA 

data = np.array([[1,2,3,2,1,2,1,1],[1,1,1,1,2,3,1,1],[1,2,2,1,2,3,1,1],[1,1,3,1,4,3,1,1],[1,3,1,4,2,3,1,1],[2,1,2,1,2,3,1,1],[1,2,2,1,2,3,1,1],[1,1,2,1,3,3,1,1],[2,1,3,1,2,3,1,1],[1,1,2,1,2,2,1,1]])

pca=PCA(n_components=2)
newData=pca.fit_transform(data)

print(newData)
