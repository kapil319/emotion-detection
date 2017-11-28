import os
import csv
import random as rn
import math
import operator 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from collections import Counter

dire =r"G:\5th sem\ee320 DSP\project\csv\all"
#print os.listdir(dire)
finaldata1=[]
finaldata=[]
for filename in os.walk(dire):
	for x in filename[2]:
		file=filename[0]+"\\"+str(x)
		with open(file,'rb') as csvfile:
			dataset=csv.reader(csvfile)
			data=list(dataset)
			data2=np.array(data).T.tolist()
			for l in data2:
				finaldata1.append(l)






rn.shuffle(finaldata1)
labelset=[]
for x in finaldata1:
	labelset.append(x[-1])
	del(x[-1])

for x in finaldata1:
	finaldata.append(x)

print labelset


'''X_train=np.array(finaldata)
X_scaled = preprocessing.scale(X_train)
finaldata=X_scaled.tolist()
finaldata=preprocessing.normalize(finaldata)'''


'''traindata=[]
testdata=[]
labeltrain=[]
labeltest=[]

l=len(finaldata)

ltrain=int(0.80*l)
ltest=l-ltrain


for x in range(ltrain):
	traindata.append(finaldata[x])
	labeltrain.append(labelset[x])

for x in range(ltest):
	testdata.append(finaldata[x+ltrain])
	labeltest.append(labelset[x+ltest])'''


'''neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(traindata,labeltrain) 

testpred=neigh.predict(testdata)'''
clf = svm.SVC()
clf.fit(finaldata, labelset)


dire =r"G:\5th sem\ee320 DSP\project\csv\test1"
#print os.listdir(dire)
finaldata1=[]
testdata=[]
labeltest=[]
testpredfinal=[]
for filename in os.walk(dire):
	for x in filename[2]:
		file=filename[0]+"\\"+str(x)
		with open(file,'rb') as csvfile:
			dataset=csv.reader(csvfile)
			data=list(dataset)
			data2=np.array(data).T.tolist()
			labeltest.append(data2[0][-1])
			for i in data2:
				del(i[-1])
			testpred=clf.predict(data2)
			print testpred
			most_common,num_most_common = Counter(testpred).most_common(1)[0]
			testpredfinal.append(most_common)

print testpredfinal
print labeltest

print accuracy_score(labeltest,testpredfinal)





