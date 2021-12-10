import pickle
import operator
import os
from os import path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


file = open("/data/movie-associations/movset/itemsets.pickle",'rb')
itemsets = pickle.load(file)
#total number of 1s slices: 2851272






idf={} #only needed to get a list of the keys. FIXME
segments_list=[]
for j in range(1,2850):#short for debug, FIXME
	DF = {}
	total=0;
	n='max'#2851272
	for i in range(j*1000,j*1000+1000):
		tokens = itemsets[i]
		for w in tokens:
			total=total+1
			if (w in DF.keys()):
				DF[w]=DF[w]+1
			else:
				DF[w] = 1
			if (w in idf.keys()):
				idf[w]=idf[w]+1
			else:
				idf[w] = 1
				#print("added term: ", w)
	segments_list.append(DF)


sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
#for key in sorted_idf:
#	logfile.write(str(key[0])+'\t'+str(key[1]))
#logfile.close()


print("finished calculating idf!")


#calculates tf-idf metrics as a measure of potential stationarity+rarity

if path.exists("tfidf_full1000.log"):
	os.remove("tfidf_full1000.log")
logfile  = open("idf_full1000.log", "a")

tf_idf={}
for key in idf:
	for DF in segments_list:
		if (key in DF.keys()):
			if (key in tf_idf.keys()):
				if DF[key]/idf[key] > tf_idf[key]:
					tf_idf[key]=DF[key]/idf[key]
			else:
				tf_idf[key]=DF[key]/idf[key]


sorted_tfidf = sorted(tf_idf.items(), key=operator.itemgetter(1),reverse=True)
#for key in sorted_tfidf:
#	logfile.write(str(key[0])+'\t'+str(key[1])+'\n')
#logfile.close()


k=0
x=np.arange(start=1000, stop=2850000, step=1000)
y1=[]
ynames=[]
print("finished calculating tfidf!")

adf={}
adf_b={}

kpssd={}
kpssd_b={}

#calculates mentions in each 1000s segment, checks for stationarity with ADF
for key in idf.keys():
	k=k+1
	if (k%1000)==0:
		print(str(k)+" keys done")
	#if k>5000:
	#	break
	y=[]
	for DF in segments_list:
		if (key in DF.keys()):
			y.append(DF[key])
		else:
			y.append(0)
	y1.append(y)
	#y_nonzero=[0 if a==0 else 1 for a in y] 
	result = adfuller(y)
	adf[key] = result
	kpsstest = kpss(y, regression='c')
	kpssd[key]=kpsstest
	nonstat=0
	if (result[1] > 0.05):
		nonstat=1
	if (result[1]<=0.005)&(result[0]>=-2.57):
		nonstat=1
	adf_b[key]=nonstat	
	#plt.plot(x, y, alpha=0.1)

#print(len(y1))
#print(ynames)

if path.exists("adf_full1000.log"):
	os.remove("adf_full1000.log")
logfile  = open("adf_full1000.log", "a")

if path.exists("adf_b_1000.log"):
	os.remove("adf_b_1000.log")
logfile_b  = open("adf_b_1000.log", "a")


if path.exists("kpss_full1000.log"):
	os.remove("kpss_full1000.log")
logfile  = open("kpss_full1000.log", "a")


for key in adf:
	logfile.write(str(key)+'\t'+str(adf[key][0])+'\t'+str(adf[key][1])+'\t'+str(adf[key][4])+'\n')
logfile.close()
for key in adf_b:
	logfile_b.write(str(key)+'\t'+str(adf_b[key])+'\n')
logfile_b.close()

#for key in kpssd:
#	logfile.write(str(key)+'\t'+str(kpssd[key][0])+'\t'+str(kpssd[key][1])+'\t'+str(kpssd[key][4])+'\n')
#logfile.close()

filename = 'idf_segments_of_1000.pickle'
outfile = open(filename,'wb')
pickle.dump(idf,outfile)
outfile.close()

filename = 'tfidf_segments_of_1000.pickle'
outfile = open(filename,'wb')
pickle.dump(tf_idf,outfile)
outfile.close()

