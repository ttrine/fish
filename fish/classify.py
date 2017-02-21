import os, sys
import random
import numpy as np
import pandas
import h5py

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import ImageDataGenerator

from fish.chunk import chunk_mask, chunk_image
from fish.sequence import train_sequencer

class ClassifierContainer:
	def __init__(self,name,model,n,optimizer,loss_weights=[1.,1.],datagen_args=dict()):
		# Set instance variables
		self.name = name
		self.n = n
		self.datagen_args = datagen_args

		# Compile model
		model.compile(optimizer=optimizer, loss=[
			"binary_crossentropy","categorical_crossentropy"], loss_weights=loss_weights)
		self.model = model
		
		# Load train data
		data = h5py.File('data/train/binary/data.h5','r')
		self.X_train = data['X_train']

		y_masks_train = data['y_masks_train'][:]
		self.y_masks_train = y_masks_train.reshape((3021,974,1732,1))
	
		y_classes_train = np.load('data/train/binary/y_classes_train.npy')
		self.y_classes_train = np_utils.to_categorical(pandas.factorize(y_classes_train, sort=True)[0])

		# Load test data
		self.X_test_chunks = np.load('data/train/binary/X_test_chunk_seqs_'+str(n)+'.npy')
		self.X_test_locations = np.load('data/train/binary/X_test_loc_seqs_'+str(n)+'.npy')

		y_test_coverage = np.load('data/train/binary/y_test_coverage_seqs_'+str(n)+'.npy')
		self.y_test_coverage = y_test_coverage.reshape((y_test_coverage.shape[0], y_test_coverage.shape[1], 1))

		self.y_classes_test = np.load('data/train/binary/y_classes_test_onehot.npy')

		# Load eval data
		eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
		self.X_eval = eval_data['X']
		self.filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

	# Returns a generator that produces sequences to train against
	def sample_gen(self,batch_size):
		seed = 1 # For reproducibility

		image_datagen = ImageDataGenerator(**self.datagen_args)
		mask_datagen = ImageDataGenerator(**self.datagen_args)

		image_gen = image_datagen.flow(self.X_train,self.y_classes_train,batch_size=batch_size,seed=seed)
		mask_gen = mask_datagen.flow(self.y_masks_train,None,batch_size=batch_size,seed=seed)

		while True:
			images, classes = image_gen.next()
			masks = mask_gen.next()
			if masks.shape != ((batch_size,974,1732,1)): # Fix unknown wrong-shape error during reshape
				continue
			masks = masks.reshape((batch_size,974,1732))

			chunk_sequences = []
			location_sequences = []
			coverage_sequences = []
			class_labels = []
			for i in range(len(images)):
				chunk_matrix = chunk_image(self.n,images[i])
				coverage_matrix = chunk_mask(self.n, chunk_matrix, masks[i])
				chunk_sequence, coverage_sequence, location_sequence = train_sequencer(chunk_matrix, coverage_matrix)
				chunk_sequences.append(chunk_sequence)
				location_sequences.append(location_sequence)
				coverage_sequences.append(coverage_sequence)
				class_labels.append(classes[i])

			chunk_sequences = pad_sequences(chunk_sequences).astype(np.float32)
			location_sequences = pad_sequences(location_sequences).astype(np.float32)
			coverage_sequences = pad_sequences(coverage_sequences).astype(np.float32)
			coverage_sequences = coverage_sequences.reshape((coverage_sequences.shape[0], coverage_sequences.shape[1], 1))
			class_labels = np.array(class_labels)

			yield [chunk_sequences, location_sequences], [coverage_sequences, class_labels]

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		train_gen = self.sample_gen(batch_size)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=([self.X_test_chunks,self.X_test_locations],[self.y_test_coverage,self.y_classes_test]), verbose=1, callbacks=[model_checkpoint])

	''' TODO. Predict class for each image. '''
	def evaluate(self,weight_file, chunk_matrices, prediction_matrices, k):
		self.model.load_weights('experiments/'+self.name+'/weights/'+weight_file)

		results = []
		for i in range(len(chunk_matrices)):
			chunk_matrix = chunk_matrices[i]
			coverage_matrix = (prediction_matrices[i] > k).astype(np.float32)
			if not np.any(coverage_matrix): # Set pr of class nofish to 1
				results.append(np.array([0., 0., 0., 0., 1., 0., 0., 0.]))
			else: # Form a 1-sequence batch and classify it
				chunk_seq, location_seq = sequencer(chunk_matrix,coverage_matrix)
				chunk_seq = chunk_seq.reshape((1,chunk_seq.shape[0],self.n,self.n,3)).astype(np.float32)
				location_seq = location_seq.reshape((1,location_seq.shape[0],2))
				if i % 50 == 0: print str(i) + " images processed by classifier."
				class_prediction = self.model.predict([chunk_seq,location_seq])[0]
				results.append(class_prediction)

		return results
