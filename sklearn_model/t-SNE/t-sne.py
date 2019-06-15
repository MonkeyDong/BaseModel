import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape

'''显示原始数据'''
n = 20  # 每行20个数字，每列20个数字
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()

'''
t-SNE的优化
在优化t-SNE方面，有很多技巧。下面5个参数会影响t-SNE的可视化效果：

perplexity 混乱度。混乱度越高，t-SNE将考虑越多的邻近点，更关注全局。因此，对于大数据应该使用较高混乱度，较高混乱度也可以帮助t-SNE拜托噪声的影响。相对而言，该参数对可视化效果影响不大。
early exaggeration factor 该值表示你期望的簇间距大小，如果太大的话（大于实际簇的间距），将导致目标函数无法收敛。相对而言，该参数对可视化效果影响较小，默认就行。
learning rate 学习率。关键参数，根据具体问题调节。
maximum number of iterations 迭代次数。迭代次数不能太低，建议1000以上。
angle (not used in exact method) 角度。相对而言，该参数对效果影响不大。
'''
