import math
import numpy as np

def overlap(c_x1,c_x2,c_y1,c_y2,a_x1,a_x2,a_y1,a_y2):
	# Returns true only if regions horizontal and vertical regions both overlap
	x_overlap = len(set(range(c_x1,c_x2)).intersection(set(range(a_x1,a_x2))))>0
	y_overlap = len(set(range(c_y1,c_y2)).intersection(set(range(a_y1,a_y2))))>0
	return x_overlap and y_overlap

def chunk_train(n,y_boxes,img,mask,filename):
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

def chunk_test(n,img):
	# Insert augmentation here

	ncol = int(math.ceil(float(1732)/float(n)))
	nrow = int(math.ceil(float(974)/float(n)))

	# Chunk up image. If chunk has coverage, it will be present once for each box it covers.
	img_chunks = []
	locations = [] # For evaluation pipeline
	for j in range(ncol):
		for i in range(nrow):
			x1 = j*n
			y1 = i*n
			x2 = ((j+1)*n)
			y2 = ((i+1)*n)
			img_chunk = img[y1:y2,x1:x2]
			if not np.any(img_chunk): 
				continue # skip all-black chunks
			if img_chunk.shape != (n,n,3): # skip the rare case in which bottom/right-most chunks are nonblack
				continue
			img_chunks.append(img_chunk)
			locations.append((i,j))
	return (img_chunks,locations)
