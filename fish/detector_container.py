import os, csv, random, math
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from preprocess import TEST_DIR

random.seed(1)

def read_boxes():
	import csv
	y_boxes_file = open("data/train/y_boxes.csv",'rb')
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

class ModelContainer:
	def __init__(self,name,model,optimizer=Adam(lr=1e-5)):
		self.name = name

		model.compile(optimizer=optimizer, loss="mse")
		self.model = model
		
		# Load raw-ish data, parceled out into splits
		print "Loading X..."
		self.X_train = np.load('data/train/X_train.npy')
		self.X_test = np.load('data/train/X_test.npy')
		self.X_eval = np.load('data/test_stg1/X.npy')

		print "Loading y..."
		self.y_masks_train = np.load('data/train/y_masks_train.npy')
		self.y_masks_test = np.load('data/train/y_masks_test.npy')
		self.y_filenames_train = np.load('data/train/y_filenames_train.npy')
		self.y_filenames_test = np.load('data/train/y_filenames_test.npy')
		self.y_classes_train = np.load('data/train/y_classes_train.npy')
		self.y_classes_test = np.load('data/train/y_classes_test.npy')
		self.y_boxes = read_boxes()

	def chunk(n):
		# Get random image and its metadata
		index = random.sample(range(len(self.X_train)),1)[0]

		img = X_train[index]
		mask = y_masks_train[index]
		filename = y_filenames_train[index].split('/')[-1]

		# Insert augmentation here

		ncol = int(math.ceil(float(1732)/float(n)))
		nrow = int(math.ceil(float(974)/float(n)))

		# Chunk up image. If chunk has coverage, it will be present once for each box it covers.
		img_chunks = []
		chunk_labels = []
		for j in range(ncol):
			for i in range(nrow):
				x1 = j*n
				y1 = i*n
				x2 = ((j+1)*n)-1
				y2 = ((i+1)*n)-1
				img_chunk = img[y1:y2,x1:x2]
				if not np.any(img_chunk): continue # skip all-black chunks
				if img_chunk.shape != (n-1,n-1,3): # skip the rare case in which bottom/right-most chunks are nonblack
					continue
				mask_chunk = mask[y1:y2,x1:x2]
				if not np.any(mask_chunk): # work is done, short-circuit the labeling
					img_chunks.append(img_chunk)
					chunk_labels.append(np.array([0,0,0,0,0]))
				else: # compute relative top-left and bottom-right bounding-box coords for each fish
					if filename in y_boxes: # just in case it isn't
						annotations = y_boxes[filename]
						for annotation in annotations:
							x_center = int(round(np.mean([x1,x2])))
							y_center = int(round(np.mean([y1,y2])))
							x_dist_tr = annotation['x1'] - x_center
							y_dist_tr = annotation['y1'] - y_center
							x_dist_br = annotation['x2'] - x_center
							y_dist_br = annotation['y2'] - y_center
							img_chunks.append(img_chunk)
							chunk_labels.append(np.array([x_dist_tr,y_dist_tr,x_dist_br,y_dist_br,1]))
					else: continue
		return (img_chunks,chunk_labels)

	def sample_gen(n,samples_per_epoch): # n: side length of chunks
		chunks = []
		labels = []
		while True:
			img_chunks,chunk_labels = self.chunk(n)
			chunks.extend(img_chunks)
			labels.extend(chunk_labels)
			if len(chunks) >= samples_per_epoch:
				# Randomize, cast, yield
				shuffle = random.sample(range(1000),1000)
				chunks = np.array(chunks)[shuffle].astype(np.float32)
				labels = np.array(labels)[shuffle]
				yield (chunks,labels)
				# Keep leftover samples for next epoch
				chunks = chunks[samples_per_epoch:len(chunks)]
				labels = labels[samples_per_epoch:len(chunks)]

	''' Trains the model according to the desired 
		specifications. '''
	def train(self, weight_file=None, n=100, nb_epoch=40, samples_per_epoch=1000):
		model_folder = 'data/models/' + self.name + '/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		print model_folder+self.name+'_{epoch:02d}-{loss:.2f}.hdf5'
		model_checkpoint = ModelCheckpoint(model_folder+self.name+'_{epoch:02d}-{loss:.2f}.hdf5', monitor='loss')
		gen = self.sample_gen(n,samples_per_epoch)
		self.model.fit_generator(gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=None, verbose=1, callbacks=[model_checkpoint])

	''' Runs the test set through the network and 
		converts the result to regulation format. '''
	def evaluate(self,weight_file,submission_name=None):# VAR NAMES HAVE CHANGED
		if submission_name is None: submission_name = weight_file.split('.hdf5')[0] + '_submission'
		model_folder = 'data/models/' + self.name + '/'

		self.model.load_weights('data/models/'+self.name+'/'+weight_file)
		predictions = self.model.predict(self.test_images, verbose=1)
		if not os.path.exists('data/models/'+self.name):
			os.makedirs('data/models/'+self.name)

		with open('data/models/'+self.name+'/'+submission_name+'.csv', 'w+') as csvfile:
			output = csv.writer(csvfile, delimiter=',')
			output.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
			filenames = [filename for filename in os.listdir(TEST_DIR) if filename.split('.')[1]!='npy']
			filenames.sort()
			for i,pred in enumerate(predictions):
				output.writerow([filenames[i]] + [str(col) for col in pred])
