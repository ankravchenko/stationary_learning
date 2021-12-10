import pickle
import operator
import os
from os import path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

file = open("/data/movie-associations/movset/itemsets.pickle",'rb')
itemsets = pickle.load(file)
#total number of 1s slices: 2851272

idf={}
if path.exists("idf_full1000.log"):
	os.remove("idf_full1000.log")
logfile  = open("idf_full1000.log", "a")

#cuts data into 1000s segments, calculates inverse "document" frequence across all of them
segments_list=[]
for j in range(1,2850):
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
for key in sorted_idf:
	logfile.write(str(key[0])+'\t'+str(key[1]))
logfile.close()


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


sorted_tfidf = sorted(tf_idf.items(), key=operator.itemgetter(1))
for key in sorted_tfidf:
	logfile.write(str(key[0])+'\t'+str(key[1])+'\n')
logfile.close()


k=0
x=np.arange(start=1000, stop=2850000, step=1000)
y1=[]
ynames=[]
print("finished calculating tfidf!")

#this is data from later stages of analysis
rarekeys=['Sunflower', 'Darts', 'Apple', 'Wasp', 'Hammock', 'Hourglass', 'Corn', 'Duck', 'Seal', 'Wind Turbine']#handpicked from between 10-20 top percent tfidf
nonstatkeys=['Swallow', 'Tree Frog', 'Afghan Hound', 'Surgeonfish', 'Avocado', 'Octopus', 'Salamander', 'Snowflake', 'Aurora', 'Mantis']#handpicked from confirmed to be nonstat by ADF test

#calculates mentions in each 1000s segment
for key in idf:
	if k==100:
		break
	if k%10==0:
		ynames.append(key)
		y=[]
		print(key)
		for DF in segments_list:
			if (key in DF.keys()):
				y.append(DF[key])
			else:
				y.append(0)
	#plt.plot(x, y, alpha=0.1)
		y1.append(np.nonzero(y)[0].tolist())
	k=k+1

#print(len(y1))
#print(ynames)


#parameters for the raster plot
colorCodes = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
						[0, 0, 0],
                        [1, 0, 0]])

offsets = [1,2,3,4,5,6,7,8,9,10]
linewidth = np.repeat(0.2, 10)


fig, ax = plt.subplots()
#ax.eventplot(y1_np, orientation="horizontal")#, color=colorCodes)# lineoffsets=offsets, linelengths=linewidth, 

ax.eventplot(y1, orientation="horizontal")#, color=colorCodes)# lineoffsets=offsets, linelengths=linewidth, 

#xnames=np.arange(start=1000, stop=2850000, step=10000)

print(y1_np.shape)
ax.set_xlim(right=2850)
xnames=np.arange(0, 2850000, 285000)
ax.xaxis.set_ticks(np.arange(0, 2850, 285))
ax.yaxis.set_ticks(np.arange(0, 10, 1))
ax.set_yticklabels(ynames, rotation='horizontal', fontsize=7)
ax.set_xticklabels(xnames, rotation='horizontal', fontsize=7)

plt.savefig("raster_1000.png")

filename = 'idf_segments_of_1000.pickle'
outfile = open(filename,'wb')
pickle.dump(idf,outfile)
outfile.close()

filename = 'tfidf_segments_of_1000.pickle'
outfile = open(filename,'wb')
pickle.dump(tf_idf,outfile)
outfile.close()

