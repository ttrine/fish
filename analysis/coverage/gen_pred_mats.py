import csv
import numpy as np
import h5py
import pandas

from keras.utils import np_utils

from keras.preprocessing.sequence import pad_sequences

from fish.detect import DetectorContainer
from fish.classify import ClassifierContainer

from fish.chunk import chunk_image, chunk_mask

from experiments.recurrent_detect_seq_response.train import construct as construct_detector
from experiments.classify_basic.train import construct as construct_classifier

from fish.sequence import detector_sequencer_inf

from theano.tensor.nnet.nnet import categorical_crossentropy

# Load data
data = h5py.File('data/train/binary/data.h5','r')
X_test = data['X_test']
X_train = data['X_train']
y_masks_test = data['y_masks_test']

y_classes_test = np.load('data/train/binary/y_classes_test.npy')
y_classes_test = np_utils.to_categorical(pandas.factorize(y_classes_test, sort=True)[0])

# Generate chunk and prediction matrices
detector = DetectorContainer('recurrent_detect_seq_response',construct_detector(256),256,'adam')
detector.model.load_weights('experiments/recurrent_detect_seq_response/weights/22-0.1232.hdf5')

predictions = []
for ind in range(len(X_train)):
	img_chunks = chunk_image(256,X_train[ind])
	chunk_seq, location_seq = detector_sequencer_inf(img_chunks)
	if ind % 50 == 0: print str(ind) + " images processed by detector."
	chunk_seq = chunk_seq.reshape((1,chunk_seq.shape[0],256,256,3)).astype(np.float32)
	location_seq = location_seq.reshape((1,location_seq.shape[0],2)).astype(np.float32)
	pred_seq = detector.model.predict([chunk_seq,location_seq])
	predictions.append(pred_seq[0][:,0])

predictions = pad_sequences(predictions, dtype=np.float32)

np.save('data/train/binary/X_test_pred_seqs_256',predictions)

# Generage coverage matrices
coverage_mats = []
for i in range(len(chunks)):
	coverage_mats.append(chunk_mask(256,chunks[i],y_masks_train[i]))

# Make chunk, location sequences according to both prediction and coverage matrices
chunk_seq_inf = []
loc_seq_inf = []
chunk_seq_lab = []
loc_seq_lab = []

for i in range(len(chunks)):
	inf_coverage = (predictions[i] > .25).astype(np.uint8)
	c, l = sequencer(chunks[i],inf_coverage)
	chunk_seq_inf.append(c)
	loc_seq_inf.append(l)
	c, l = sequencer(chunks[i], coverage_mats[i])
	chunk_seq_lab.append(c)
	loc_seq_lab.append(l)

classifier = ClassifierContainer('classify_basic',construct_classifier(256),256,'adam')
classifier.model.load_weights('experiments/classify_basic/weights/63-0.3345.hdf5')

preds = []
for i in range(len(classifier.X_test_chunk_seqs)):
	chunk_seq = classifier.X_test_chunk_seqs[i][np.any(classifier.X_test_chunk_seqs[i],(1,2,3))]
	location_seq = classifier.X_test_loc_seqs[i][np.any(classifier.X_test_chunk_seqs[i],(1,2,3))]
	chunk_seq = chunk_seq.reshape((1,chunk_seq.shape[0],256,256,3)).astype(np.float32)
	location_seq = location_seq.reshape((1,location_seq.shape[0],2))
	pred = classifier.model.predict([chunk_seq,location_seq])
	preds.append(pred)
	if i % 50 == 0: print str(i) + " images processed by classifier."

preds = np.array(preds)
preds[np.where(preds==1)] = .9999

l = categorical_crossentropy(preds,classifier.y_classes_test).eval()
np.mean(l)