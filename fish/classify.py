import os, sys, csv
import random
import numpy as np
import pandas
import h5py

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import ImageDataGenerator

from fish.chunk import chunk_mask, chunk_image

class ClassifierContainer:
	def __init__(self,name,model,n,optimizer,datagen_args=dict(),loss_weights=[1.,1.],callbacks=[]):
		# Set instance variables
		self.name = name
		self.n = n
		self.datagen_args = datagen_args

		# Compile model
		model.compile(optimizer=optimizer, loss=["binary_crossentropy","categorical_crossentropy"], loss_weights=loss_weights)

		self.model = model
		
		# Load train data
		self.X_train = np.load('data/train/binary/X_train_2.npy')

		y_masks_train = np.load('data/train/binary/y_masks_train_2.npy')
		self.y_masks_train = y_masks_train.reshape((3021,487,866,1))
	
		y_classes_train = np.load('data/train/binary/y_classes_train.npy')
		self.y_classes_train = np_utils.to_categorical(pandas.factorize(y_classes_train, sort=True)[0])

		# Load test data
		self.X_test = np.load('data/train/binary/X_test_2.npy')
		self.y_test_coverage = np.load('data/train/binary/y_test_coverage_mats_'+str(n)+'.npy')
		self.y_classes_test = np.load('data/train/binary/y_classes_test_onehot.npy')

		# Load eval data
		eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
		self.X_eval = eval_data['X_eval']
		self.filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

		self.callbacks = callbacks

	# Returns a generator that produces sequences to train against
	def sample_gen(self,batch_size):
		# We employ the bootstrap method to train several models and average their results.
		bootstrap_inds = np.random.choice(len(self.X_train), len(self.X_train))
		bootstrap_samps = self.X_train[bootstrap_inds]
		bootstrap_classes = self.y_classes_train[bootstrap_inds]
		bootstrap_masks = self.y_masks_train[bootstrap_inds]

		seed = 1

		image_datagen = ImageDataGenerator(**self.datagen_args)
		mask_datagen = ImageDataGenerator(**self.datagen_args)

		image_gen = image_datagen.flow(bootstrap_samps,bootstrap_classes,batch_size=batch_size,seed=seed)
		mask_gen = mask_datagen.flow(bootstrap_masks,None,batch_size=batch_size,seed=seed)

		while True:
			images, classes = image_gen.next()
			masks = mask_gen.next()
			if masks.shape != ((batch_size,487, 866,1)): # Fix unknown wrong-shape error during reshape
				continue
			masks = masks.reshape((batch_size,487, 866))

			coverage_matrices = []
			class_labels = []
			for i in range(len(images)):
				chunk_matrix = chunk_image(self.n,images[i], 2)
				coverage_matrix = chunk_mask(self.n, chunk_matrix, masks[i], 2)
				
				coverage_matrices.append(coverage_matrix)
				class_labels.append(classes[i])

			coverage_matrices = np.array(coverage_matrices)
			class_labels = np.array(class_labels)

			yield images, [coverage_matrices, class_labels]

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		self.callbacks.append(model_checkpoint)
		train_gen = self.sample_gen(batch_size)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=(self.X_test,[self.y_test_coverage,self.y_classes_test]), verbose=1, callbacks=self.callbacks)

	''' TODO. Predict class for each image. '''
	def evaluate(self,weight_file,clip=False):
		self.model.load_weights('experiments/'+self.name+'/weights/'+weight_file)

		print "Running inference..."

		predictions = self.model.predict(self.X_eval, verbose=True)[1]

		if clip:
			predictions = np.clip(predictions,0.02, 0.98, out=None)

		f = file('experiments/'+self.name+'/submission.csv','wb')
		w = csv.writer(f)
		w.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
		rows = [[filename] for filename in self.filenames_eval]
		[rows[i].extend(list(predictions[i])) for i in range(1000)]
		w.writerows(rows)
		f.close()
		print "Done. Wrote experiments/"+self.name+"/submission.csv."
