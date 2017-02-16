import os, sys
import csv, random
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
from keras import backend as K

from fish.chunk import chunk_mask, chunk_image
from fish.sequence import detector_sequencer

class DetectorContainer:
	def __init__(self,name,model,n,optimizer=Adam(lr=1e-5),datagen_args=dict()):
		self.name = name

		model.compile(optimizer=optimizer, loss="binary_crossentropy")
		self.model = model
		
		# Load raw-ish data, parceled out into splits
		data = h5py.File('data/train/binary/data.h5','r')
		self.X_train = data['X_train']
		y_masks_train = data['y_masks_train'][:]
		self.y_masks_train = y_masks_train.reshape((3021,974,1732,1))

		self.X_test_chunks = np.load('data/train/binary/X_test_det_chunk_seqs_'+str(n)+'.npy')
		self.X_test_locations = np.load('data/train/binary/X_test_det_loc_seqs_'+str(n)+'.npy')
		y_test_coverage = np.load('data/train/binary/y_test_det_coverage_seqs_'+str(n)+'.npy')
		self.y_test_coverage = y_test_coverage.reshape((y_test_coverage.shape[0], y_test_coverage.shape[1], 1))

		self.n = n

		self.datagen_args = datagen_args
		
		eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
		self.X_eval = eval_data['X']
		self.filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

	def sample_gen(self,batch_size):
		seed = 1 # For reproducibility

		image_datagen = ImageDataGenerator(**self.datagen_args)
		mask_datagen = ImageDataGenerator(**self.datagen_args)

		image_gen = image_datagen.flow(self.X_train,None,batch_size=batch_size,seed=seed)
		mask_gen = mask_datagen.flow(self.y_masks_train,None,batch_size=batch_size,seed=seed)

		while True:
			images = image_gen.next()
			masks = mask_gen.next()
			if masks.shape != ((batch_size,974,1732,1)): # Fix unknown wrong-shape error during reshape
				continue
			masks = masks.reshape((batch_size,974,1732))

			chunk_sequences = []
			location_sequences = []
			coverage_sequences = []
			for i in range(len(images)):
				chunk_matrix = chunk_image(self.n,images[i])
				coverage_matrix = chunk_mask(self.n, chunk_matrix, masks[i])
				chunk_sequence, coverage_sequence, location_sequence = detector_sequencer(chunk_matrix, coverage_matrix)
				chunk_sequences.append(chunk_sequence)
				coverage_sequences.append(coverage_sequence)
				location_sequences.append(location_sequence)

			chunk_sequences = pad_sequences(chunk_sequences).astype(np.float32)
			location_sequences = pad_sequences(location_sequences).astype(np.float32)
			coverage_sequences = pad_sequences(coverage_sequences).astype(np.float32)
			coverage_sequences = coverage_sequences.reshape((coverage_sequences.shape[0], coverage_sequences.shape[1], 1))

			yield [chunk_sequences, location_sequences], coverage_sequences

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		train_gen = self.sample_gen(batch_size)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=([self.X_test_chunks, self.X_test_locations],self.y_test_coverage), verbose=1, callbacks=[model_checkpoint])

	''' Produce matrix of scores for each chunk in each evaluation image. '''
	def evaluate(self,weight_file):
		self.model.load_weights('experiments/'+self.name+'/weights/'+weight_file)

		chunks = []
		predictions = []
		for ind in range(len(self.X_eval)):
			img_chunks = chunk_image(self.n,self.X_eval[ind])
			img_predictions = np.zeros(img_chunks.shape)
			if ind % 50 == 0: print str(ind) + " images processed by detector."
			for i in range(img_chunks.shape[0]):
				for j in range(img_chunks.shape[1]):
					if img_chunks[i,j] is None: continue # Don't attempt inference on all-black chunks
					img_predictions[i,j] = self.model.predict(img_chunks[i,j].reshape(1,self.n,self.n,3).astype(np.float32))
			chunks.append(img_chunks)
			predictions.append(img_predictions)
		return chunks,predictions
