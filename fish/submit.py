import csv
import numpy as np
import h5py

from keras.preprocessing.sequence import pad_sequences

from fish.detect import DetectorContainer
from fish.classify import ClassifierContainer

from fish.chunk import chunk_image

from experiments.recurrent_detect_seq_response.train import construct as construct_detector
from experiments.classify_soft_thresh.train import construct as construct_classifier

from fish.sequence import detector_sequencer_inf

eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
X_eval = eval_data['X']
filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

# Get predictions from detector

detector = DetectorContainer('recurrent_detect_seq_response',construct_detector(256),256,'adam')
detector.model.load_weights('experiments/recurrent_detect_seq_response/weights/22-0.1232.hdf5')

predictions = []
for ind in range(len(X_eval)):
	img_chunks = chunk_image(256,X_eval[ind])
	chunk_seq, location_seq = detector_sequencer_inf(img_chunks)
	if ind % 50 == 0: print str(ind) + " images processed by detector."
	chunk_seq = chunk_seq.reshape((1,chunk_seq.shape[0],256,256,3)).astype(np.float32)
	location_seq = location_seq.reshape((1,location_seq.shape[0],2)).astype(np.float32)
	pred_seq = detector.model.predict([chunk_seq,location_seq])
	predictions.append(pred_seq[0][:,0])

predictions = pad_sequences(predictions, dtype=np.float32)

# Feed chunks, predictions, locations into classifier

classifier = ClassifierContainer('classify_soft_thresh',construct_classifier(256),256,'adam')
classifier.model.load_weights('experiments/classify_soft_thresh/weights/23-0.4759.hdf5')

class_predictions = []
for ind in range(len(X_eval)):
	img_chunks = chunk_image(256,X_eval[ind])
	chunk_seq, location_seq = detector_sequencer_inf(img_chunks)
	pred_seq = predictions[ind][-chunk_seq.shape[0]:]
	if ind % 50 == 0: print str(ind) + " images processed by classifier."
	chunk_seq = chunk_seq.reshape((1,chunk_seq.shape[0],256,256,3)).astype(np.float32)
	pred_seq = pred_seq.reshape((1,pred_seq.shape[0],1))
	location_seq = location_seq.reshape((1,location_seq.shape[0],2)).astype(np.float32)
	class_prediction = classifier.model.predict([chunk_seq,pred_seq,location_seq])
	class_predictions.append(class_prediction)

submission = file('experiments/classify_soft_thresh/submission.csv','wb')
w = csv.writer(submission)
w.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])

for i in range(1000):
	row = [filenames_eval[i]]
	row.extend(list(class_predictions[i]))
	w.writerow(row)

submission.close()
