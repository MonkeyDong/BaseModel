import numpy as np
import os

kkey = []
paths = ['datas/ACt','datas/NMt','datas/PPt','datas/ACs','datas/NMs','datas/PPs']

def get_keys(path):
	alls = []
	files = os.listdir(path)
	for f in files:
		if f == '.DS_Store':
			continue
		ff = np.load(path+'/'+f)
		ll = []
		dics = {}
		for j,i in enumerate(ff.T):
			ss = set(i)
			if ss != {0}:
				ll = list(ss)
				ll.remove(0)
				dics[j] = ll
		keys = list(dics.keys())
		alls.append(keys)
	return alls

for path in paths:
	kkey = kkey + get_keys(path)

np.save("keys.npy",kkey)#alls中存着每一张谱图有mz的通道的序号数组

###############################################
#找出所有向量共同存在的mz通道坐标
'''
arr = []
tmp = alls[0]
for l in alls[1:]:
	for n in l:
		if n in tmp:
			arr.append(n)
	tmp = arr
	arr = []

choose = tmp
np.save("choose.npy",choose)
print(choose)
'''
###############################################
#统计所有mz通道的出现频率
dics = {}
for i in kkey:
	for j in i:
		if not (j in dics.keys()):
			dics[j] = 1
		if j in dics.keys():
			dics[j] += 1

'''
value = []
for v in dics.values():
	value.append(v)
va = sorted(value)

tt = 0
for i in va:
	if i > 20:
		tt += 1

'''
lists = []
for k in dics.keys():
	if dics[k] > 20:
		lists.append(k)

np.save("choose.npy",lists)



