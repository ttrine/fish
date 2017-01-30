import math, h5py
import numpy as np

from fish.detector_container import read_boxes, overlap

data = h5py.File('data/train/binary/data.h5','r')

X_test = data['X_test']
y_masks_test = data['y_masks_test']
y_boxes = read_boxes()
y_filenames_test = np.load('data/train/binary/y_filenames_test.npy','r')

def chunk(img,mask,filename,n):
	ncol = int(math.ceil(float(1732)/float(n)))
	nrow = int(math.ceil(float(974)/float(n)))
	img_chunks = []
	chunk_labels = []
	for j in range(ncol):
		for i in range(nrow):
			x1 = j*n
			y1 = i*n
			x2 = ((j+1)*n)
			y2 = ((i+1)*n)
			img_chunk = img[y1:y2,x1:x2]
			if not np.any(img_chunk): continue
			if img_chunk.shape != (n,n,3): continue
			mask_chunk = mask[y1:y2,x1:x2]
			pct_coverage = float(sum(sum((mask_chunk>0).astype(np.uint8))))/float(n*n)
			if pct_coverage < .001: # not enough of fish in chunk, label as no fish
				img_chunks.append(img_chunk)
				chunk_labels.append(np.array([0,0,0,0,0]))
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
				else: continue
	return (img_chunks,chunk_labels)

def compute_and_write(n):
	chunks = []
	labels = []
	for i in range(len(X_test)):
		filename = y_filenames_test[i].split('/')[-1]
		img_chunks,chunk_labels = chunk(X_test[i],y_masks_test[i],filename,n)
		chunks.extend(img_chunks)
		labels.extend(chunk_labels)

	chunks = np.array(chunks)
	labels = np.array(labels)

	np.save("data/train/binary/X_test_chunks_"+str(n),chunks)
	np.save("data/train/binary/y_test_chunks_"+str(n),labels)

if __name__ == '__main__':
	import sys
	compute_and_write(int(sys.argv[1]))