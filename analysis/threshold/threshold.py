import numpy as np

from fish.detector_container import ModelContainer
from experiments.detect_batch_more_feats import construct,sgd

import csv

experiment = "detect_batch_more_feats"
weights = "detect_batch_more_feats_124-0.1845.hdf5"
n = 256
loss = sgd

# Load trained model, generate test predictions
model = ModelContainer('experiment',construct(n),n,loss)
model.model.load_weights("experiments/"+experiment+"/"+weights)

predictions = model.model.predict(model.X_test, verbose=1)

# Generate threshold data
y_test = model.y_test[:,-1]
thresh = np.zeros((len(y_test),2),np.float32)
thresh[:,0] = y_test
thresh[:,1] = predsictions[:,0]
f = file("analysis/threshold/"+experiment+".csv",'wb')
w = csv.writer(f)
w.writerows(thresh)
f.close()
