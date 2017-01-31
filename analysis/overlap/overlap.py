import math, csv

from fish.detector_container import ModelContainer # Just to load data

# Global constants
n = 64
ncol = int(math.ceil(float(1732)/float(n)))
nrow = int(math.ceil(float(974)/float(n)))

# Chunks image and reports % coverage of each chunk
def pct_coverage(img,mask):
	img_overlaps = []
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
			img_overlaps.append(pct_coverage)
	return img_overlaps

# Collect up all the % coverages from each image
model = ModelContainer('irrelevant',construct(256),256,sgd)
overlaps = []
for i in range(len(model.X_train)):
	overlaps.extend(pct_coverage(model.X_train[i],model.y_masks_train[i]))

# Convert to columns, write to file
overlaps = [[overlap] for overlap in overlaps]

o = file("analysis/misc/train_overlap_percentages.csv",'wb')
o_write = csv.writer(o)
o_write.writerows(overlaps)
o.close()
