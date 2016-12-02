import csv
import datetime
import numpy as np, numpy.random
import itertools
import os 


imagefiles = os.listdir("test_stg1")


def sumbit_results():
	timenow = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
	name = timenow +'.csv'
	with open(name, 'w') as csvfile:
		output = csv.writer(csvfile, delimiter=',')
		output.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
		for i in range(len(imagefiles)):
			temp = [imagefiles[i]]
			fakeResults = np.random.dirichlet(np.ones(8),size=1).tolist()
			fakeMerged = list(itertools.chain.from_iterable(fakeResults))
			fakeString = [str(j) for j in fakeMerged]
			temp = temp + fakeString
			output.writerow(temp)






			



