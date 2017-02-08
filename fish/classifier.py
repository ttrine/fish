import os, sys
import random
import numpy as np
import pandas
import h5py

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from fish.chunk import chunk_mask, chunk_image
from fish.sequencer import sequence

class ClassifierContainer:
	def __init__(self,name,model,n,optimizer):
		self.name = name

		model.compile(optimizer=optimizer, loss="categorical_crossentropy")
		self.model = model
		
		# Load raw-ish data, parceled out into splits
		data = h5py.File('data/train/binary/data.h5','r')
		self.X_train = data['X_train']
		self.y_masks_train = data['y_masks_train']
		self.y_filenames_train = np.load('data/train/binary/y_filenames_train.npy')

		# Convert class labels to 1-hot schema
		y_classes_train = np.load('data/train/binary/y_classes_train.npy')
		self.y_classes_train = np_utils.to_categorical(pandas.factorize(y_classes_train, sort=True)[0])

		try: # Test data must be precomputed
			self.X_test_chunk_seqs = np.load('data/train/binary/X_test_chunk_seqs_'+str(n)+'.npy')
			self.X_test_loc_seqs = np.load('data/train/binary/X_test_loc_seqs_'+str(n)+'.npy')
			self.y_classes_test = np.load('data/train/binary/y_test_classes_onehot_fish_'+str(n)+'.npy')
		except:
			print "Precomputed test data not found for that chunk size."
			sys.exit()

		self.n = n
		
		eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
		self.X_eval = eval_data['X']
		self.filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

	# Returns a generator that produces sequences to train against
	def sample_gen(self,batch_size):
		random.seed(1) # For reproducibility
		chunk_sequences = []
		location_sequences = []
		class_labels = []
		while True:
			index = random.sample(range(len(self.X_train)),1)[0]

			class_label = self.y_classes_train[index]
			chunk_matrix = chunk_image(self.n,self.X_train[index])
			coverage_matrix = chunk_mask(self.n,chunk_matrix,self.y_masks_train[index])
			if not np.any(coverage_matrix): continue # No images without fish please
			chunk_sequence, location_sequence = sequence(chunk_matrix, coverage_matrix)
			class_labels.append(class_label)
			chunk_sequences.append(chunk_sequence)
			location_sequences.append(location_sequence)
			if len(chunk_sequences) == batch_size:
				chunk_sequences = pad_sequences(chunk_sequences).astype(np.float32)
				location_sequences = pad_sequences(location_sequences).astype(np.float32)
				class_labels = np.array(class_labels)
				yield [chunk_sequences, location_sequences], class_labels
				chunk_sequences = []
				location_sequences = []
				class_labels = []

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		train_gen = self.sample_gen(batch_size)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=([self.X_test_chunk_seqs,self.X_test_loc_seqs],self.y_classes_test), verbose=1, callbacks=[model_checkpoint])
