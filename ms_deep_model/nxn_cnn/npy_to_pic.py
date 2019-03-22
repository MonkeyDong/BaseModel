import os
import numpy as np

#Z-score标准化
def z_score(x):
	x = (x - np.average(x))/np.std(x)
	return x


#################################################
files = os.listdir("datas_mz/ACt")
nn = len(files)
Xt1 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xt1[m,:] = np.load("datas_mz/ACt/"+n)[:2304]

Xt1 = z_score(Xt1)
labels_t1 = [0]*nn
#################################################
files = os.listdir("datas_mz/NMt")
nn = len(files)
Xt2 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xt2[m,:] = np.load("datas_mz/NMt/"+n)[:2304]

Xt2 = z_score(Xt2)
labels_t2 = [1]*nn
#################################################
files = os.listdir("datas_mz/PPt")
nn = len(files)
Xt3 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xt3[m,:] = np.load("datas_mz/PPt/"+n)[:2304]

Xt3 = z_score(Xt3)
labels_t3 = [2]*nn
#################################################
files = os.listdir("datas_mz/ACs")
nn = len(files)
Xs1 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xs1[m,:] = np.load("datas_mz/ACs/"+n)[:2304]

Xs1 = z_score(Xs1)
labels_t4 = [0]*nn
#################################################
files = os.listdir("datas_mz/NMs")
nn = len(files)
Xs2 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xs2[m,:] = np.load("datas_mz/NMs/"+n)[:2304]

Xs2 = z_score(Xs2)
labels_t5 = [1]*nn
#################################################
files = os.listdir("datas_mz/PPs")
nn = len(files)
Xs3 = np.zeros((nn,2304))
for m,n in enumerate(files):
	Xs3[m,:] = np.load("datas_mz/PPs/"+n)[:2304]

Xs3 = z_score(Xs3)
labels_t6 = [2]*nn
#################################################
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
labels_test = labels_t4 + labels_t5 + labels_t6

np.save("save_npy/pic_train.npy",X_train)
np.save("save_npy/pic_test.npy",X_test)
np.save("save_npy/labels_train",labels_train)
np.save("save_npy/labels_test",labels_test)

