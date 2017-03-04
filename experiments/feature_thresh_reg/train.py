import sys
sys.setrecursionlimit(10000)

import numpy as np

from keras import backend as K
from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Lambda, Dropout, Activation, SpatialDropout2D
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from keras.callbacks import Callback

from keras.regularizers import l2

from fish.layers import SpecialBatchNormalization
from fish.classify import ClassifierContainer

# Lambda function for a differentiable layer that regularizes classifier input
	# according to the detector
def fishy_features(x, tied_to):
	weights = tied_to.weights[0].reshape((1,1,1,256))
	fish_repeat = np.repeat(np.repeat(np.repeat(weights,x.shape[0],0),16,1),28,2)
	return x * fish_repeat

# Utility function to apply padding, convolution, bn, relu,
	# spatial dropout, and max pooling respectively.
def reg_conv(name, x, k, n, p, pad=True, bn=False, pool=2):
	if pad: x = ZeroPadding2D((n-2, n-2))(x)
	x = Convolution2D(k, n, n, name=name, W_regularizer=l2())(x)
	if bn: x = BatchNormalization()(x)
	x = Activation('relu')(x)
	if pool: x = MaxPooling2D(pool_size=(pool, pool))(x)

	return x

def construct():
	imgs = Input(shape=(487, 866, 3))
	batch = SpecialBatchNormalization()(imgs)

	# Root. Shared CNN for learning representations 
	# 		common to detection and classification.
	stem_1 = reg_conv("stem_1", batch, 32, 5, .25)
	stem_2 = reg_conv("stem_2", stem_1, 64, 5, .25, pool=4)
	stem_3 = reg_conv("stem_3", stem_2, 64, 5, .25, bn=True)
	stem_4 = reg_conv("stem_4", stem_3, 128, 5, .3)
	stem_5 = reg_conv("stem_5", stem_4, 256, 3, .4, bn=True, pool=False)

	# Detector. Approximates image's coverage matrix. We use weights from this layer 
	##			to restrict the classifier input to only features related to fish.
	conv_coverage = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=l2())
	pred_mat = conv_coverage(stem_5)
	pred_mat = Reshape((16,28),name="coverage")(pred_mat)

	# Classifier. Infers fish type.
	fishy_feats = Lambda(fishy_features, arguments={'tied_to': conv_coverage})(stem_5)
	fishy_feats = Activation('relu')(fishy_feats)

	class_1 = reg_conv("class_1", fishy_feats, 256, 3, .4, bn=True)
	class_2 = reg_conv("class_2", class_1, 256, 3, .4)
	class_3 = reg_conv("class_3", class_2, 256, 3, .4, bn=True)
	class_4 = reg_conv("class_4", class_3, 256, 3, .4)
	
	fcn = Flatten()(class_4)

	dense_1 = Dense(256, name="dense_1")(fcn)
	dense_1 = Activation('relu')(dense_1)
	dense_1 = Dropout(.5)(dense_1)

	dense_2 = Dense(256, name="dense_2")(dense_1)
	dense_2 = BatchNormalization(mode=1)(dense_2)
	dense_2 = Activation('relu')(dense_2)
	dense_2 = Dropout(.5)(dense_2)

	class_vec = Dense(8, activation='softmax', name="class")(dense_2)

	return Model(input=imgs,output=[pred_mat,class_vec])

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(),32,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
