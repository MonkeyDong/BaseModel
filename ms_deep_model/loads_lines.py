import numpy as np
import cv2
import os

files = os.listdir('lines')

N_all = []
T_all = []
ND = {}
TD = {}
dd = []


def listsort(list):
	kk=[]
	for i in list:
		kk.append([int(i[-11]),list.index(i),i])
	kk.sort()
	out = [i[2] for i in kk]
	return out
	

for i in files:
	if i[5] == 'N':
		if not (i[2:5] in ND):
			ND[i[2:5]] = []
			ND[i[2:5]].append(i)
			continue
		if (i[2:5] in ND) and len(ND[i[2:5]]) < 5: #6-1
			ND[i[2:5]].append(i)
			continue
		if (i[2:5] in ND) and len(ND[i[2:5]]) == 5:
			ND[i[2:5]].append(i)
			dd = listsort(ND[i[2:5]])
			N_all.append(dd)
			del ND[i[2:5]]
			continue
		else:
			continue
	if i[5] == 'T':
		if not (i[2:5] in TD):
			TD[i[2:5]] = []
			TD[i[2:5]].append(i)
			continue
		if (i[2:5] in TD) and len(TD[i[2:5]]) < 5:
			TD[i[2:5]].append(i)
			continue
		if (i[2:5] in TD) and len(TD[i[2:5]]) == 5:
			TD[i[2:5]].append(i)
			dd = listsort(TD[i[2:5]])
			T_all.append(dd)
			del TD[i[2:5]]
			continue
		else:
			continue

print('N_all:',N_all)
print('T_all:',T_all)

all1 = T_all
all2 = N_all
############################################################

ll = np.array([4,5,9,12])

def load_data(file_path,ll):
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		bars.append(cc)
	return bars


#bar = [start,end,S,mz]
def add(ma,bar):
	rts = int(round((bar[0])/q))
	rte = int(round((bar[1])/q))
	if rte > 2000 or rte < 0:
		return ma
	mzs = int(round((bar[3])/p))
	if mzs > 2099 or mzs < 0:
		return ma
	SS = bar[2]
	for i in range(rte-rts):
		ma[rts][mzs] = SS + ma[rts+i][mzs]
	return ma


p = 1#质荷比的缩放比例
q = 0.05#保留时间的缩放比例

rt = 100 #保留时间维度的矩阵长度
mz = 2100 #质荷比维度的矩阵长度

rt = int(rt/q)
mz = int(mz/p)

mm = len(all1)
results = []
res = np.zeros([rt,mz],dtype=float)
ms = np.array([])
ns = np.array([])
tt = 0

for k in all1:
	for l in k:
		bbar = load_data(os.path.join('lines',l),ll)
		for ii in bbar:
			res = add(res,ii)
	
	np.save('datas/Tumor/ms%d.npy' % (tt),res)
	#cv2.imwrite('T_train/ms%d.jpg' % (tt), res)
	res = np.zeros([rt,mz],dtype=float)
	tt = tt+1


mm = len(all2)
results = []
res = np.zeros([rt,mz],dtype=float)
ms = np.array([])
ns = np.array([])
tt = 0

for k in all2:
	for l in k:
		bbar = load_data(os.path.join('lines',l),ll)
		for ii in bbar:
			res = add(res,ii)
	
	np.save('datas/Normal/ms%d.npy' % (tt),res)
	#cv2.imwrite('T_train/ms%d.jpg' % (tt), res)
	res = np.zeros([rt,mz],dtype=float)
	tt = tt+1











