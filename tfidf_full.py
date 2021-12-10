import pickle
import operator
import os
from os import path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt



file = open("/data/movie-associations/movset/itemsets.pickle",'rb')
itemsets = pickle.load(file)

#2851272

idf={}
if path.exists("idf_full.log"):
	os.remove("idf_full.log")
logfile  = open("idf_full.log", "a")

segments_list=[]

for j in range(1,285):#285
	DF = {}
	total=0;
	n='max'#2851272
	for i in range(j*10000,j*10000+10000):
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
				print("added term: ", w)
	segments_list.append(DF)


sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))

for key in sorted_idf:
	logfile.write(str(key[0])+'\t'+str(key[1]))
logfile.close()


print("finished calculating idf!")

if path.exists("tfidf_full.log"):
	os.remove("tfidf_full.log")
logfile  = open("tfidf_full.log", "a")

tf_idf={}


for key in idf:
	for DF in segments_list:
		if (key in DF.keys()):
			if (key in tf_idf.keys()):
				if DF[key]/idf[key] > tf_idf[key]:
					tf_idf[key]=DF[key]/idf[key]
			else:
				tf_idf[key]=DF[key]/idf[key]


sorted_tfidf = sorted(tf_idf.items(), key=operator.itemgetter(1))

for key in sorted_tfidf:
	logfile.write(str(key[0])+'\t'+str(key[1])+'\n')
logfile.close()


print("finished calculating tfidf!")




