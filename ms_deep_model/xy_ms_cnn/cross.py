import numpy as np
import xlrd
import os

workbook1 = xlrd.open_workbook('TPD_training_samples_lable V20181016.xlsx')
booksheet1 = workbook1.sheet_by_index(0)  #用索引取第一个sheet
workbook2 = xlrd.open_workbook('TPD_IPX0001444002.xlsx')
booksheet2 = workbook2.sheet_by_index(0) 

dics1 = {}
rows = booksheet1.nrows
for i in range(1,rows):
	key_ = booksheet1.cell_value(i,2)
	if key_[:2] == "Mo":
		continue
	if key_[:2] == "po":
		continue
	if key_ == "":
		continue
	cell_ = booksheet1.cell_value(i,3)
	dics1[key_] = cell_ #1256个样本


dics2 = {}
rows = booksheet2.nrows
for i in range(1,rows):
	key_ = booksheet2.cell_value(i,1)
	if key_[:2] == "Mo":
		continue
	if key_[:2] == "po":
		continue
	if key_ == "":
		continue
	cell_ = booksheet2.cell_value(i,2)
	dics2[key_] = cell_ #539个样本


data1 = {}
for d in dics1.keys():
	if d[:-1] in data1.keys():
		data1[d[:-1]].append(d)
	else:
		data1[d[:-1]] = []
		data1[d[:-1]].append(d) #399个病例

data2 = {}
for d in dics2.keys():
	if d[:-1] in data2.keys():
		data2[d[:-1]].append(d)
	else:
		data2[d[:-1]] = []
		data2[d[:-1]].append(d) #180个病例


nn = 0
for i in data1.keys():
     nn += len(data1[i])#1259

mm = 0
for i in data2.keys():
     mm += len(data2[i])#539


def splits(datas,n):
	data = np.array(datas)
	index_1=np.random.choice(data.shape[0],n,replace=False)
	da1=data[index_1]
	index_2=np.arange(data.shape[0])
	index_2=np.delete(index_2,index_1)
	da2=data[index_2]
	return da1,da2

dd1,dd2 = splits(list(data1.keys()),39)
dd3,dd4 = splits(list(data2.keys()),16)

ddd1 = []
ddd2 = []
ddd3 = []
ddd4 = []


for n in data1.keys():
	if n in dd1:
		for m in data1[n]:
			ddd1.append(m)

for n in data1.keys():
	if n in dd2:
		for k in data1[n]:
			ddd2.append(k)

for n in data2.keys():
	if n in dd3:
		for m in data2[n]:
			ddd3.append(m)

for n in data2.keys():
	if n in dd4:
		for k in data2[n]:
			ddd4.append(k)

f1 = []
for i in ddd1:
	f1.append(dics1[i]+"out.txt")

f2 = []
for i in ddd2:
	f2.append(dics1[i]+"out.txt")

f3 = []
for i in ddd3:
	f3.append(dics2[i]+"out.txt")

f4 = []
for i in ddd4:
	f4.append(dics2[i]+"out.txt")

#少了几个数据
np.save("save_npy/file1.npy",f1)#106
np.save("save_npy/file2.npy",f2)#1150
np.save("save_npy/file3.npy",f3)#48
np.save("save_npy/file4.npy",f4)#491