import numpy as np

from fish.detector_container import ModelContainer
from experiments.detect_batch_more_feats import construct,sgd

# Load trained model
model = ModelContainer('detect_batch_more_feats',construct(256),256,sgd)
model.model.load_weights("data/models/detect_batch_more_feats/detect_batch_more_feats_124-0.1845.hdf5")

# Get chunks
chunks=[]
chunk_filenames=[]
for i in range(len(model.X_eval)):
	img_chunks = model.chunk_eval(model.X_eval[i])
	chunks.extend(img_chunks)
	chunk_filenames.extend(np.tile(model.filenames_eval[i],len(img_chunks)))

# Perform inference
chunks = np.array(chunks)
predictions = model.model.predict(chunks, verbose=1)

# Dump it all out
np.save("data/exploratory/inference_dump/fish_detector_test/eval_chunks.npy",chunks)
np.save("data/exploratory/inference_dump/fish_detector_test/eval_chunk_filenames.npy",chunk_filenames)
np.save("data/exploratory/inference_dump/fish_detector_test/eval_predictions.npy",predictions)

## Misc.

# Generate test set predictions
predictions = model.model.predict(model.X_test, verbose=1)
np.save("data/models/detect_batch_more_feats/test_predictions.npy",predictions)

# Generate CSV for ROC curve
y_test = model.y_test[:,-1]
roc_data = np.zeros((len(y_test),2),np.float32)
roc_data[:,0] = y_test
preds = np.load("data/models/detect_batch_more_feats/test_predictions.npy")
roc_data[:,1] = preds[:,0]
import csv
train_roc = file("data/exploratory/detect_batch_more_feats.csv",'wb')
roc_write = csv.writer(train_roc)
roc_write.writerows(roc_data)
train_roc.close()
