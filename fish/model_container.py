import os
import numpy as np
import csv

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from preprocess import TEST_DIR

class ModelContainer:
	def __init__(self,name,model,preprocess,optimizer=Adam(lr=1e-5)):
		self.name = name
		self.preprocess = preprocess

		model.compile(optimizer=optimizer, loss="categorical_crossentropy")
		self.model = model

		train_images = np.load('data/train/X_train.npy')
		train_labels = np.load('data/train/y_train.npy')
		val_images = np.load('data/train/X_valid.npy')
		val_labels = np.load('data/train/y_valid.npy')

		test_images = np.load('data/test_stg1/X_all.npy')

		train_images = preprocess(train_images).astype('float32')
		val_images = preprocess(val_images).astype('float32')
		test_images = preprocess(test_images).astype('float32')

		mean = np.mean(train_images)
		std = np.std(train_images)

		train_images -= mean
		train_images /= std

		self.train_images = train_images
		self.train_labels = train_labels
		self.val_images = val_images
		self.val_labels = val_labels

		self.test_images = test_images

	''' Trains the model according to the desired 
		specifications. '''
	def train(self, weight_file=None, nb_epoch=40, batch_size=32):
		model_folder = 'data/models/' + self.name + '/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		print model_folder+self.name+'_{epoch:02d}-{loss:.2f}.hdf5'
		model_checkpoint = ModelCheckpoint(model_folder+self.name+'_{epoch:02d}-{loss:.2f}.hdf5', monitor='loss')

		self.model.fit(self.train_images, self.train_labels, batch_size=batch_size, nb_epoch=nb_epoch, 
			validation_data=(self.val_images,self.val_labels), verbose=1, shuffle=True, callbacks=[model_checkpoint])

	def evaluate(self,weight_file,submission_name=None):
		if submission_name is None: submission_name = weight_file.split('.hdf5')[0] + '_submission'
		model_folder = 'data/models/' + self.name + '/'

		predictions = self.model.predict(self.test_images, verbose=1)
		if not os.path.exists('data/models/'+self.name):
			os.makedirs('data/models/'+self.name)

		with open('data/models/'+self.name+'/'+submission_name+'.csv', 'w+') as csvfile:
			output = csv.writer(csvfile, delimiter=',')
			output.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
			filenames = [filename for filename in os.listdir(TEST_DIR) if filename.split('.')[1]!='npy']
			for i,pred in enumerate(predictions):
				output.writerow([filenames[i]] + [str(col) for col in pred])
