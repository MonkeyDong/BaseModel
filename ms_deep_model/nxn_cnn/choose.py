import numpy as np
import os

lists = np.load("choose.npy")
paths = ['datas/ACt','datas/NMt','datas/PPt','datas/ACs','datas/NMs','datas/PPs']


def get_vecs(path):
	files = os.listdir(path)
	tt = 0
	for f in files:
		if f == '.DS_Store':
			continue
		arr_all = []
		ff = np.load(path+'/'+f)
		ll = []
		dics = {}
		for j,i in enumerate(ff.T):
			ss = set(i)
			if ss != {0}:
				ll = list(ss)
				ll.remove(0)
				dics[j] = ll
		for ii in lists:
			res = np.zeros(2)
			if ii not in dics.keys():
				res = res
			else:
				if len(dics[ii]) == 1:
					res[0] = dics[ii][0]
				if len(dics[ii]) > 2:
					res[:] = sorted(dics[ii],reverse = True)[:2]		
			arr_all = arr_all + res.tolist()
		np.save("datas_mz/"+path[-3:]+"/"+path[-3:]+"%d.npy" % (tt),arr_all)
		tt += 1	
	

for path in paths:
	get_vecs(path)




