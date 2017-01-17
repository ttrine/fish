import numpy as np
import random, math

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

# Load raw-ish data
print "Loading X..."
X = np.load('data/train/X.npy')
print "Loading y_masks..."
y_masks = np.load('data/train/y_masks.npy')
y_filenames = np.load('data/train/y_filenames.npy')
y_boxes = read_boxes()

# Set global constants
random.seed(1)
K = 10 # number of images to get
N = 100 # size of chunk side

# Get K images and their metadata
indices = random.sample(range(3777),K)

k_imgs = X[indices]
k_masks = y_masks[indices]
k_filenames = [filename.split('/')[-1] for filename in y_filenames[indices]]

# Insert augmentation here

# Chunk up images. If chunk has coverage, it will be present once for each box it covers.
img_chunks = []
chunk_labels = []

ncol = int(math.ceil(float(1732)/float(N)))
nrow = int(math.ceil(float(974)/float(N)))

for ind in range(K):
	img = k_imgs[ind]
	mask = k_masks[ind]
	for j in range(ncol):
		for i in range(nrow):
			x1 = j*100
			y1 = i*100
			x2 = ((j+1)*100)-1
			y2 = ((i+1)*100)-1
			img_chunk = img[y1:y2,x1:x2]
			if not np.any(img_chunk): continue # skip all-black chunks
			if img_chunk.shape != (N-1,N-1,3): # skip the rare case in which bottom/right-most chunks are nonblack
				continue
			mask_chunk = mask[y1:y2,x1:x2]
			if not np.any(mask_chunk): # work is done, short-circuit the labeling
				img_chunks.append(img_chunk)
				chunk_labels.append([0,0,0,0,0])
			else: # compute relative top-left and bottom-right bounding-box coords for each fish
				filename = k_filenames[ind]
				if filename in y_boxes: # just in case
					annotations = y_boxes[filename]
					for annotation in annotations:
						x_center = int(round(np.mean([x1,x2])))
						y_center = int(round(np.mean([y1,y2])))
						x_dist_tr = annotation['x1'] - x_center
						y_dist_tr = annotation['y1'] - y_center
						x_dist_br = annotation['x2'] - x_center
						y_dist_br = annotation['y2'] - y_center
						img_chunks.append(img_chunk)
						chunk_labels.append([x_dist_tr,y_dist_tr,x_dist_br,y_dist_br,1])
				else: continue
for chunk in img_chunks: print chunk
# Insert normalization here