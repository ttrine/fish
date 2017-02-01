import numpy as np

from fish.detector_container import ModelContainer
from experiments.detect_batch_more_feats.train import construct,sgd

import csv

def predict_and_dump(experiment,weights,n,loss=sgd):
	# Load trained model, generate test predictions
	model = ModelContainer('experiment',construct(n),n,loss)
	model.model.load_weights("experiments/"+experiment+"/weights/"+weights)

	predictions = model.model.predict(model.X_test, verbose=1)

	# Generate threshold data
	y_test = model.y_test[:,-1]
	thresh = np.zeros((len(y_test),2),np.float32)
	thresh[:,0] = y_test
	thresh[:,1] = predictions[:,0]
	f = file("analysis/threshold/"+experiment+".csv",'wb')
	w = csv.writer(f)
	w.writerows(thresh)
	f.close()

if __name__ == '__main__':
	import sys # experiment_name, weight_file, n

	predict_and_dump(sys.argv[1],sys.argv[2],int(sys.argv[3]))
