# Dumps masks into 4-column form, one each for the X,Y 
# of top-left and bottom-right coords.

import numpy as np

y_train_localizer = np.load("data/train/y_train_localizer.npy")
y_valid_localizer = np.load("data/train/y_valid_localizer.npy")

TL_train = np.zeros((len(y_train_localizer),4))
TL_valid = np.zeros((len(y_valid_localizer),4))

for i,mask in enumerate(y_train_localizer):
	# Get (x,y) for top-left and bottom-right coords, resp.
	if len(np.where(sum(mask)>0)[0]) == 0:
		x_top = 256 # WRONG
		x_bot = 256
	else: 
		x_top = np.min(np.where(sum(mask)>0))
		x_bot = np.max(np.where(sum(mask)>0))

	if len(np.where(sum(np.transpose(mask))>0)[0]) == 0:
		y_top = 256
		y_bot = 256
	else:
		y_top = np.min(np.where(sum(np.transpose(mask))>0))
		y_bot = np.max(np.where(sum(np.transpose(mask))>0))

	TL_train[i,0] = x_top
	TL_train[i,1] = y_top
	TL_train[i,2] = x_bot
	TL_train[i,3] = y_bot

# for i,mask in enumerate(y_valid_localizer):
	# # Get (x,y) for top-left and bottom-right coords, resp.
	# if len(np.where(sum(mask)>0)[0]) == 0:
	# 	x_top = 256
	# 	x_bot = 256
	# else: 
	# 	x_top = np.min(np.where(sum(mask)>0))
	# 	x_bot = np.max(np.where(sum(mask)>0))

	# if len(np.where(sum(np.transpose(mask))>0)[0]) == 0:
	# 	y_top = 256
	# 	y_bot = 256
	# else:
	# 	y_top = np.min(np.where(sum(np.transpose(mask))>0))
	# 	y_bot = np.max(np.where(sum(np.transpose(mask))>0))

	# TL_valid[i,0] = x_top
	# TL_valid[i,1] = y_top
	# TL_valid[i,2] = x_bot
	# TL_valid[i,3] = y_bot

# np.save("data/train/y_train_coord",TL_train)
# np.save("data/train/y_valid_coord",TL_valid)