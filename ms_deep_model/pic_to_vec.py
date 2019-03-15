import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

#样本不平衡  #重采样算法
pass

#labels -> one_hot
def one_hot(y):
	lb = LabelBinarizer()
	lb.fit(y)
	yy = lb.transform(y)
	return yy

#从保存着转化完成的数据的文件夹中导入数据

files = os.listdir("datas/Tumor")
nn = len(files)
X1 = np.zeros((nn, 256, 21))

for m,n in enumerate(files):
	ms = np.load("datas/Tumor/"+n)
	ms = ms[100:1500,300:1100]

	xs = []
	xx = np.zeros((800))
	ss = []
	for i,j in enumerate(ms):
		xx = xx + j
		if i%200 == 0:
			for j in range(3):
				ss = xx[j*256:(j+1)*256]
				xs.append(ss)

	m_s = np.array(xs).transpose(1,0)
	X1[m,:,:] = m_s

#X_ (nn,256,21) 数据转化为长度为256的一维向量，通道数为21

#归一化
X1 = X1/100000000
'''
def standardize(train, test):
	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test

'''
pass
#打乱  #shuffle()
pass

labels_1 = [0]*nn

####################################################################

files = os.listdir("datas/Normal")
nn = len(files)
X2 = np.zeros((nn, 256, 21))

for m,n in enumerate(files):
	ms = np.load("datas/Normal/"+n)
	ms = ms[100:1500,300:1100]

	xs = []
	xx = np.zeros((800))
	ss = []
	for i,j in enumerate(ms):
		xx = xx + j
		if i%200 == 0:
			for j in range(3):
				ss = xx[j*256:(j+1)*256]
				xs.append(ss)

	m_s = np.array(xs).transpose(1,0)
	X2[m,:,:] = m_s

#归一化
X2 = X2/100000000
pass
#打乱  #shuffle()
pass

labels_2 = [1]*nn

######################################################################

#随机噪声组
Xr = rand_noise = np.random.normal(size=(nn,256,21))
labels_3 = [2]*nn

######################################################################
#训练集合并
#X = X1 + X2 + X3 + ...
#后面还需要加上多分类
pass
X11 = X1.tolist()
X22 = X2.tolist()
Xr = Xr.tolist()
XX = [i for i in X11]+[j for j in X22]+[k for k in Xr]
X_train = np.array(XX)

labels_train = labels_1 + labels_2 + labels_3

# Train/Validation Split
#后面还要加上Test
pass

X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)

y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)

np.save("save_npy/x_train.npy",X_tr)
np.save("save_npy/x_ver.npy",X_vld)
np.save("save_npy/y_train.npy",y_tr)
np.save("save_npy/y_ver.npy",y_vld)

pass
#X_test = 
#labels_test = 
#y_test = one_hot(labels_test)


