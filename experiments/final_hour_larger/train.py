from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.classify import ClassifierContainer

def construct():
	imgs = Input(shape=(487, 866, 3))
	batch = BatchNormalization()(imgs)
	batch = Dropout(.2)(batch)

	# Shared CNN for learning representations common to detection and classification.
	conv1 = ZeroPadding2D((3, 3))(batch)
	conv1 = Convolution2D(16, 5, 5, activation='relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = ZeroPadding2D((3, 3))(conv1)
	conv2 = Convolution2D(64, 5, 5, activation='relu')(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv2 = Dropout(.5)(conv2)

	conv3 = ZeroPadding2D((3, 3))(conv2)
	conv3 = Convolution2D(64, 5, 5, activation='relu')(conv3)
	conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = ZeroPadding2D((3, 3))(conv3)
	conv4 = Convolution2D(256, 5, 5, activation='relu')(conv4)
	conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	conv4 = Dropout(.5)(conv4)

	conv5 = ZeroPadding2D((1, 1))(conv4)
	conv5 = Convolution2D(256, 3, 3, activation='relu')(conv5)
	
	conv_gate = ZeroPadding2D((1, 1))(conv5)
	conv_gate = Convolution2D(256, 3, 3, activation='sigmoid')(conv_gate)
	conv_gate = Dropout(.5)(conv_gate)

	conv_coverage = ZeroPadding2D((1, 1))(conv_gate)
	conv_coverage = Convolution2D(1, 3, 3, activation='sigmoid')(conv_coverage)
	conv_coverage = MaxPooling2D(pool_size=(2, 2))(conv_coverage)

	# Shave off channel dimension
	pred_mat = Reshape((16,28),name="coverage")(conv_coverage)

	gated_feats = merge([conv_gate, conv5], mode='mul')

	conv_class1 = ZeroPadding2D((1, 1))(gated_feats)
	conv_class1 = Convolution2D(256, 3, 3, activation='relu')(conv_class1)
	conv_class1 = MaxPooling2D(pool_size=(2, 2))(conv_class1)
	conv_class1 = Dropout(.5)(conv_class1)

	conv_class2 = ZeroPadding2D((1, 1))(conv_class1)
	conv_class2 = Convolution2D(256, 3, 3, activation='relu')(conv_class2)
	conv_class2 = MaxPooling2D(pool_size=(2, 2))(conv_class2)

	conv_class3 = ZeroPadding2D((1, 1))(conv_class2)
	conv_class3 = Convolution2D(512, 3, 3, activation='relu')(conv_class3)
	conv_class3 = MaxPooling2D(pool_size=(2, 2))(conv_class3)
	conv_class3 = Dropout(.5)(conv_class3)

	conv_class4 = ZeroPadding2D((1, 1))(conv_class3)
	conv_class4 = Convolution2D(256, 3, 3, activation='relu')(conv_class4)
	conv_class4 = MaxPooling2D(pool_size=(2, 2))(conv_class4)

	fcn_class = Flatten()(conv_class4)
	fcn_class = Dropout(.5)(fcn_class)
	fcn_class = Dense(256, activation='relu')(fcn_class)
	fcn_class = Dropout(.25)(fcn_class)
	fcn_class = Dense(256, activation='relu')(fcn_class)
	class_vec = Dense(8, activation='softmax', name="class")(fcn_class)

	return Model(input=imgs,output=[pred_mat,class_vec])

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(),32,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
