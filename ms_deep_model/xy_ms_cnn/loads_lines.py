import numpy as np
import cv2
import xlrd
import os

file11 = os.listdir('TPD1XIC')
file22 = os.listdir('TPD2XIC')

workbook = xlrd.open_workbook('TPD_training_samples_lable V20181016.xlsx')
booksheet = workbook.sheet_by_index(0)         #用索引取第一个sheet

dics = {}
rows = booksheet.nrows
for i in range(rows):
	key_ = booksheet.cell_value(i,3)
	cell_ = booksheet.cell_value(i,2)
	dics[key_] = cell_

f1 = np.load("save_npy/file1.npy")
ff1 = f1.tolist()

arr = []
for i in range(len(ff1)):
	if ff1[i] not in file11:
		arr.append(ff1[i])
for i in arr:
	ff1.remove(i)


AA = []
MM = []
CC = []
PP = []
NN = []
for i in ff1:
	if not (i[:-7] in dics.keys()):
		continue 
	if dics[i[:-7]] == "":
		continue
	if dics[i[:-7]][:2] == "po":
		continue
	if dics[i[:-7]][:2] == "Mo":
		continue
	if dics[i[:-7]][0] == "A":
		AA.append(i)
	if dics[i[:-7]][0] == "M":
		MM.append(i)
	if dics[i[:-7]][0] == "C":
		CC.append(i)
	if dics[i[:-7]][0] == "P":
		PP.append(i)
	if dics[i[:-7]][0] == "N":
		NN.append(i)


############################################################

ll = np.array([9,12])

def load_data(file_path,ll):
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		bars.append(cc)
		bars = sorted(bars, key=lambda arr: arr[0],reverse=True)
	return bars


#######################################################
mm = len(NN+MM)
tt = 0

for k in NN+MM:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/NMs/nm1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/nm%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(AA+CC)
tt = 0

for k in AA+CC:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/ACs/ac1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/ac%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(PP)
tt = 0

for k in PP:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/PPs/pp1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/p%d.jpg' % (tt), res)
	tt = tt+1

######################################################################
######################################################################

f2 = np.load("save_npy/file2.npy")	
ff2 = f2.tolist()

arr = []
for i in range(len(ff2)):
	if ff2[i] not in file11:
		arr.append(ff2[i])
for i in arr:
	ff2.remove(i)


AA = []
MM = []
CC = []
PP = []
NN = []
for i in ff2:
	if not (i[:-7] in dics.keys()):
		continue 
	if dics[i[:-7]] == "":
		continue
	if dics[i[:-7]][:2] == "po":
		continue
	if dics[i[:-7]][:2] == "Mo":
		continue
	if dics[i[:-7]][0] == "A":
		AA.append(i)
	if dics[i[:-7]][0] == "M":
		MM.append(i)
	if dics[i[:-7]][0] == "C":
		CC.append(i)
	if dics[i[:-7]][0] == "P":
		PP.append(i)
	if dics[i[:-7]][0] == "N":
		NN.append(i)


#######################################################
mm = len(NN+MM)
tt = 0

for k in NN+MM:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/NMt/nm1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/nm%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(AA+CC)
tt = 0

for k in AA+CC:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/ACt/ac1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/ac%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(PP)
tt = 0

for k in PP:
	bbar = load_data(os.path.join('TPD1XIC',k),ll)
	np.save('datas/PPt/p1%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/p%d.jpg' % (tt), res)
	tt = tt+1


######################################################################
######################################################################
workbook2 = xlrd.open_workbook('TPD_IPX0001444002.xlsx')
booksheet2 = workbook2.sheet_by_index(0) 

dics2 = {}
rows = booksheet2.nrows
for i in range(1,rows):
	key_ = booksheet2.cell_value(i,2)
	cell_ = booksheet2.cell_value(i,1)
	dics2[key_] = cell_ 


f3 = np.load("save_npy/file3.npy")
ff3 = f3.tolist()

arr = []
for i in range(len(ff3)):
	if ff3[i] not in file22:
		arr.append(ff3[i])
for i in arr:
	ff3.remove(i)

AA = []
MM = []
CC = []
PP = []
NN = []
for i in ff3:
	if not (i[:-7] in dics2.keys()):
		continue 
	if dics2[i[:-7]] == "":
		continue
	if dics2[i[:-7]][:2] == "po":
		continue
	if dics2[i[:-7]][:2] == "Mo":
		continue
	if dics2[i[:-7]][0] == "A":
		AA.append(i)
	if dics2[i[:-7]][0] == "M":
		MM.append(i)
	if dics2[i[:-7]][0] == "C":
		CC.append(i)
	if dics2[i[:-7]][0] == "P":
		PP.append(i)
	if dics2[i[:-7]][0] == "N":
		NN.append(i)


#######################################################
mm = len(NN+MM)
tt = 0

for k in NN+MM:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/NMs/nm2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/nm%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(AA+CC)
tt = 0

for k in AA+CC:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/ACs/ac2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/ac%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(PP)
tt = 0

for k in PP:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/PPs/p2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/p%d.jpg' % (tt), res)
	tt = tt+1


#########################################################
#########################################################

f4 = np.load("save_npy/file4.npy")
ff4 = f4.tolist()

arr = []
for i in range(len(ff4)):
	if ff4[i] not in file22:
		arr.append(ff4[i])
for i in arr:
	ff4.remove(i)

AA = []
MM = []
CC = []
PP = []
NN = []
for i in ff4:
	if not (i[:-7] in dics2.keys()):
		continue 
	if dics2[i[:-7]] == "":
		continue
	if dics2[i[:-7]][:2] == "po":
		continue
	if dics2[i[:-7]][:2] == "Mo":
		continue
	if dics2[i[:-7]][0] == "A":
		AA.append(i)
	if dics2[i[:-7]][0] == "M":
		MM.append(i)
	if dics2[i[:-7]][0] == "C":
		CC.append(i)
	if dics2[i[:-7]][0] == "P":
		PP.append(i)
	if dics2[i[:-7]][0] == "N":
		NN.append(i)


#######################################################
mm = len(NN+MM)
tt = 0

for k in NN+MM:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/NMt/nm2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/nm%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(AA+CC)
tt = 0

for k in AA+CC:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/ACt/ac2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/ac%d.jpg' % (tt), res)
	tt = tt+1

#######################################################
mm = len(PP)
tt = 0

for k in PP:
	bbar = load_data(os.path.join('TPD2XIC',k),ll)
	np.save('datas/PPt/p2%d.npy' % (tt),bbar)
	#cv2.imwrite('T_train/p%d.jpg' % (tt), res)
	tt = tt+1

#######################################################

