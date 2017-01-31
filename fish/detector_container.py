import os, sys
import csv, random, math
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

def read_boxes():
	import csv
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

def overlap(c_x1,c_x2,c_y1,c_y2,a_x1,a_x2,a_y1,a_y2):
	# Returns true only if regions horizontal and vertical regions both overlap
	x_overlap = len(set(range(c_x1,c_x2)).intersection(set(range(a_x1,a_x2))))>0
	y_overlap = len(set(range(c_y1,c_y2)).intersection(set(range(a_y1,a_y2))))>0
	return x_overlap and y_overlap

def chunk(n,y_boxes,img,mask,filename):
	# Insert augmentation here

	ncol = int(math.ceil(float(1732)/float(n)))
	nrow = int(math.ceil(float(974)/float(n)))

	# Chunk up image. If chunk has coverage, it will be present once for each box it covers.
	img_chunks = []
	chunk_labels = []
	filenames = [] # For easier debugging
	for j in range(ncol):
		for i in range(nrow):
			x1 = j*n
			y1 = i*n
			x2 = ((j+1)*n)
			y2 = ((i+1)*n)
			img_chunk = img[y1:y2,x1:x2]
			if not np.any(img_chunk): continue # skip all-black chunks
			if img_chunk.shape != (n,n,3): # skip the rare case in which bottom/right-most chunks are nonblack
				continue
			mask_chunk = mask[y1:y2,x1:x2]
			pct_coverage = float(sum(sum((mask_chunk>0).astype(np.uint8))))/float(n*n)
			if pct_coverage < .001: # not enough of fish in chunk, label as no fish
				img_chunks.append(img_chunk)
				chunk_labels.append(np.array([0,0,0,0,0]))
				filenames.append(filename)
			else: # compute relative top-left and bottom-right bounding-box coords for each fish
				if filename in y_boxes: # just in case it isn't
					annotations = y_boxes[filename]
					for annotation in annotations:
						if overlap(x1,x2,y1,y2,annotation['x1'],annotation['x2'],annotation['y1'],annotation['y2']):
							# only add annotation if fish is in chunk
							x_center = int(round(np.mean([x1,x2])))
							y_center = int(round(np.mean([y1,y2])))
							x_dist_tr = annotation['x1'] - x_center
							y_dist_tr = annotation['y1'] - y_center
							x_dist_br = annotation['x2'] - x_center
							y_dist_br = annotation['y2'] - y_center
							img_chunks.append(img_chunk)
							chunk_labels.append(np.array([x_dist_tr,y_dist_tr,x_dist_br,y_dist_br,1]))
							filenames.append(filename)
				else: continue
	return (img_chunks,chunk_labels,filenames)

class ModelContainer:
	def __init__(self,name,model,n,optimizer=Adam(lr=1e-5)):
		self.name = name

		model.compile(optimizer=optimizer, loss="binary_crossentropy")
		self.model = model
		
		# Load raw-ish data, parceled out into splits
		data = h5py.File('data/train/binary/data.h5','r')
		self.X_train = data['X_train']
		self.y_masks_train = data['y_masks_train']
		self.y_filenames_train = np.load('data/train/binary/y_filenames_train.npy')
		self.y_classes_train = np.load('data/train/binary/y_classes_train.npy')
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

	def sample_gen(self,batch_size,X,y_masks,y_filenames): # Yield only coverage indicator
		def sample_gen(batch_size,X,y_masks,y_filenames): # Yield full BB label
			random.seed(1) # For reproducibility
			chunks = []
			labels = []
			filenames = []
			while True:
				# Get random image and its labels
				index = random.sample(range(len(X)),1)[0]
				img = X[index]
				mask = y_masks[index]
				filename = y_filenames[index].split('/')[-1]
				img_chunks,chunk_labels,filename = chunk(self.n,self.y_boxes,img,mask,filename)
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

		gen = sample_gen(batch_size,X,y_masks,y_filenames)
		while True:
			chunks, labels = gen.next()
			isfish_labels = labels[:,-1].astype(np.float32)
			yield (chunks,isfish_labels)

	def train(self, weight_file=None, nb_epoch=40, batch_size=500,samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder+self.name+weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		train_gen = self.sample_gen(batch_size,self.X_train,self.y_masks_train,self.y_filenames_train)
		
		# Convert test labels to coverage only
		y_test = self.y_test[:,-1].astype(np.float32)

		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=(self.X_test,y_test), verbose=1, callbacks=[model_checkpoint])

	''' Not yet functional. '''
	def evaluate(self,weight_file,submission_name=None):
		if submission_name is None: submission_name = weight_file.split('.hdf5')[0] + '_submission'
		model_folder = 'data/models/' + self.name + '/'

		self.model.load_weights('data/models/'+self.name+'/'+weight_file)
		predictions = self.model.predict(self.test_images, verbose=1)

		with open('data/models/'+self.name+'/'+submission_name+'.csv', 'w+') as csvfile:
			output = csv.writer(csvfile, delimiter=',')
			output.writerow(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
			filenames = [filename for filename in os.listdir("data/test_stg1/binary") if filename.split('.')[1]!='npy']
			filenames.sort()
			for i,pred in enumerate(predictions):
				output.writerow([filenames[i]] + [str(col) for col in pred])
