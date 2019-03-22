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

#Z-score标准化
def z_score(x):
	x = (x - np.average(x))/np.std(x)
	return x

#从保存着转化完成的数据的文件夹中导入数据

files = os.listdir("datas_mz/ACt")
nn = len(files)
Xt1 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/ACt/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xt1[m,:,:,:] = ms


#归一化
Xt1 = z_score(Xt1)
labels_t1 = [0]*nn

####################################################################

files = os.listdir("datas_mz/NMt")
nn = len(files)
Xt2 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/NMt/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xt2[m,:,:,:] = ms


Xt2 = z_score(Xt2)
labels_t2 = [1]*nn

######################################################################


files = os.listdir("datas_mz/PPt")
nn = len(files)
Xt3 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/PPt/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xt3[m,:,:,:] = ms

Xt3 = z_score(Xt3)

labels_t3 = [2]*nn
######################################################################
files = os.listdir("datas_mz/ACs")
nn = len(files)
Xs1 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/ACs/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xs1[m,:,:,:] = ms

Xs1 = z_score(Xs1)

labels_s1 = [0]*nn
#################################################################

files = os.listdir("datas_mz/NMs")
nn = len(files)
Xs2 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/NMs/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xs2[m,:,:,:] = ms

Xs2 = z_score(Xs2)

labels_s2 = [1]*nn

######################################################################

files = os.listdir("datas_mz/PPs")
nn = len(files)
Xs3 = np.zeros((nn, 896, 2, 1))

for m,n in enumerate(files):
	ms = np.zeros((896,2,1))	
	xs = np.load("datas_mz/PPs/"+n)
	xs = np.array(sorted(xs, key=lambda arr: arr[1],reverse=True))
	xxs = np.zeros((896,2))		
	if len(xs) > 896:
		xxs = xs[:896,:]
	elif len(xs) < 796:
		nn = nn -1
		continue
	else:
		xxs[:len(xs),:] = xs
	ms[:,:,0] = np.array(xxs)
	Xs3[m,:,:,:] = ms

Xs3 = z_score(Xs3)

labels_s3 = [2]*nn
################################################################
#训练集合并

X1 = Xt1.tolist()
X2 = Xt2.tolist()
X3 = Xt3.tolist()
X4 = Xs1.tolist()
X5 = Xs2.tolist()
X6 = Xs3.tolist()
Xtt = [i for i in X1]+[j for j in X2]+[k for k in X3]
Xss = [i for i in X4]+[j for j in X5]+[k for k in X6]

X_train = np.array(Xtt)
X_test = np.array(Xss)

labels_train = labels_t1 + labels_t2 + labels_t3
labels_test = labels_s1 + labels_s2 + labels_s3

np.save("save_npy/Xg_X_train.npy",X_train)
np.save("save_npy/Xg_X_test.npy",X_test)
np.save("save_npy/Xg_labels_train.npy",labels_train)
np.save("save_npy/Xg_labels_test.npy",labels_test)

X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, test_size=0.1,random_state = 123)

y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)

np.save("save_npy/x_train.npy",X_tr)
np.save("save_npy/x_ver.npy",X_vld)
np.save("save_npy/X_test.npy",X_test)
np.save("save_npy/y_train.npy",y_tr)
np.save("save_npy/y_ver.npy",y_vld)
np.save("save_npy/y_test.npy",y_test)




