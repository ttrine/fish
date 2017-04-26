import math
import numpy as np

''' Returns coverage matrix. '''
def chunk_mask(n,chunk_matrix,mask,resize_factor = 1):
	ncol = int(math.ceil(float(1732/resize_factor)/float(n)))
	nrow = int(math.ceil(float(974/resize_factor)/float(n)))
	coverage_matrix = np.zeros((nrow,ncol), dtype=np.uint8)

	for j in range(ncol):
		for i in range(nrow):
			if chunk_matrix[i,j] is None: continue # Exclude coverage over all-black chunks
			x1 = j*n
			y1 = i*n
			x2 = ((j+1)*n)
			y2 = ((i+1)*n)
			mask_chunk = mask[y1:y2,x1:x2]
			pct_coverage = float(sum(sum((mask_chunk>0).astype(np.uint8))))/float(n*n)
			if pct_coverage < .05: # not enough of fish in chunk, label as no fish
				coverage_matrix[i,j] = 0
			else:
				coverage_matrix[i,j] = 1
	return coverage_matrix

''' Returns chunk matrix. '''
def chunk_image(n,img,resize_factor = 1):
	ncol = int(math.ceil(float(1732/resize_factor)/float(n)))
	nrow = int(math.ceil(float(974/resize_factor)/float(n)))
	img_chunks = np.ndarray((nrow,ncol), dtype=object)

	for i in range(nrow):
		for j in range(ncol):
			x1 = j*n
			y1 = i*n
			x2 = ((j+1)*n)
			y2 = ((i+1)*n)
			img_chunk = img[y1:y2,x1:x2]
			if not np.any(img_chunk): continue # For memory efficiency
			if img_chunk.shape != (n,n,3): # Embed in correctly sized array
				embed = np.zeros((n,n,3),dtype=np.uint8)
				embed[0:img_chunk.shape[0],0:img_chunk.shape[1]] = img_chunk
				img_chunk = embed
			img_chunks[i,j] = img_chunk
	return img_chunks
