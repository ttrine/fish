import os, sys
import csv, random
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from fish.chunk import *

def read_boxes():
	y_boxes_file = open("data/train/binary/y_boxes.csv",'rb')
	box_reader = csv.reader(y_boxes_file)
	box_reader.next() # skip header
	y_boxes = [{'filename': box[0].split('/')[-1],'x1':int(box[1]),'y1':int(box[2]),'x2':int(box[3]),'y2':int(box[4])} for box in box_reader]
	y_boxes_file.close()
	d = {}
	for box in y_boxes:
		if box['filename'] in d:
			d[box['filename']].append({'x1':box['x1'],'y1':box['y1'],'x2':box['x2'],'y2':box['y2']})
		else:
			d[box['filename']]=[{'x1':box['x1'],'y1':box['y1'],'x2':box['x2'],'y2':box['y2']}]
	y_boxes = d
	return y_boxes

class DetectorContainer:
	def __init__(self,name,model,n,optimizer=Adam(lr=1e-5)):
		self.name = name

		model.compile(optimizer=optimizer, loss="binary_crossentropy")
		self.model = model
		
		# Load raw-ish data, parceled out into splits
		data = h5py.File('data/train/binary/data.h5','r')
		self.X_train = data['X_train']
		self.y_masks_train = data['y_masks_train']
		self.y_boxes = read_boxes()

		try: # Test data must be precomputed
			self.X_test = np.load('data/train/binary/X_test_chunks_'+str(n)+'.npy')
			self.y_test = np.load('data/train/binary/y_test_chunks_'+str(n)+'.npy')
		except:
			print "Precomputed test data not found for that chunk size."
			sys.exit()

		self.n = n
		
		eval_data = h5py.File("data/test_stg1/binary/eval_data.h5",'r')
		self.X_eval = eval_data['X']
		self.filenames_eval = np.load("data/test_stg1/binary/y_filenames.npy")

	def sample_gen(self,batch_size): # Yield only coverage indicator
		def sample_gen(batch_size): # Yield full BB label
			random.seed(1) # For reproducibility
			chunks = []
			labels = []
			filenames = []
			while True:
				# Get random image and its labels
				index = random.sample(range(len(self.X_train)),1)[0]
				img = self.X_train[index]
				mask = self.y_masks_train[index]
				filename = self.y_filenames_train[index].split('/')[-1]
				img_chunks,chunk_labels,filename = chunk_detector(self.n,self.y_boxes,img,mask,filename)
				chunks.extend(img_chunks)
				labels.extend(chunk_labels)
				filenames.extend(filename)
				if len(chunks) >= batch_size:
					# Randomize, cast, yield
					shuffle = random.sample(range(len(chunks)),len(chunks))
					sample_chunks = np.array(chunks)[shuffle].astype(np.float32)[0:batch_size]
					sample_labels = np.array(labels)[shuffle][0:batch_size]
					sample_filenames = np.array(filenames)[shuffle][0:batch_size]
					yield (sample_chunks,sample_labels)
					# Keep leftover samples for next epoch
					chunks = list(chunks[batch_size:len(chunks)])
					labels = list(labels[batch_size:len(labels)])

		gen = sample_gen(batch_size)
		while True:
			chunks, labels = gen.next()
			isfish_labels = labels[:,-1].astype(np.float32)
			yield (chunks,isfish_labels)

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		train_gen = self.sample_gen(batch_size)
		
		# Convert test labels to coverage only
		y_test = self.y_test[:,-1].astype(np.float32)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=(self.X_test,y_test), verbose=1, callbacks=[model_checkpoint])

	''' Produce matrix of scores for each chunk in each evaluation image. '''
	def evaluate(self,weight_file):
		self.model.load_weights('experiments/'+self.name+'/weights/'+weight_file)

		chunks = []
		predictions = []
		for ind in range(len(self.X_eval)):
			img_chunks = chunk_image(self.n,self.X_eval[ind])
			img_predictions = np.zeros(img_chunks.shape)
			print "Evaluating image " + str(ind)
			for i in range(img_chunks.shape[0]):
				for j in range(img_chunks.shape[1]):
					if img_chunks[i,j] is None: continue # Don't attempt inference on all-black chunks
					img_predictions[i,j] = self.model.predict(img_chunks[i,j].reshape(1,self.n,self.n,3).astype(np.float32))
			chunks.append(img_chunks)
			predictions.append(img_predictions)
		return chunks,predictions
