import numpy as np

from keras import backend as K
from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Lambda, Dropout
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from keras.callbacks import Callback

from keras.regularizers import WeightRegularizer
from keras.constraints import nonneg

from fish.layers import SpecialBatchNormalization
from fish.classify import ClassifierContainer

# Set up infrastructure for eliminating nonfish features.
fish_feats = K.ones((32,16,28,256))

class UpdateFeatureMask(Callback):
    def __init__(self, fish_feats):
    	self.fish_feats = fish_feats

    def on_batch_begin(self, index, logs={}):
    	filter_weights = np.array(self.model.layers[-4].weights[0].eval())
    	is_fish_feat = (filter_weights > 0).astype(np.float32).reshape(1,1,1,256)
    	is_fish_repeat = np.repeat(np.repeat(np.repeat(is_fish_feat,logs['size'],0),16,1),28,2)
    	K.set_value(self.fish_feats, is_fish_repeat)

def construct():
	imgs = Input(shape=(487, 866, 3))
	batch = SpecialBatchNormalization()(imgs)

	# Root. Shared CNN for learning representations 
	# 		common to detection and classification.
	conv1 = ZeroPadding2D((3, 3))(batch)
	conv1 = Convolution2D(16, 5, 5, activation='relu', name="stem_1")(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = ZeroPadding2D((3, 3))(conv1)
	conv2 = Convolution2D(32, 5, 5, activation='relu', name="stem_2")(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = ZeroPadding2D((3, 3))(conv2)
	conv3 = Convolution2D(64, 5, 5, activation='relu', name="stem_3")(conv3)
	conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = ZeroPadding2D((3, 3))(conv3)
	conv4 = Convolution2D(128, 5, 5, activation='relu', name="stem_4")(conv4)
	conv4 = MaxPooling2D(pool_size=(4, 4))(conv4)

	conv5 = ZeroPadding2D((1, 1))(conv4)
	conv5 = Convolution2D(256, 3, 3, activation='relu', name="stem_5")(conv5)
	
	# Detector. Approximates image's coverage matrix. We use weights from this layer 
	##			to restrict the classifier input to only features related to fish.
	conv_coverage = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=WeightRegularizer(l1=.01,l2=.01))(conv5)
	pred_mat = Reshape((16,28),name="coverage")(conv_coverage)

	# Classifier. Infers fish type.
	selected_feats = Lambda(lambda x: fish_feats * x)(conv5)

	conv_class1 = ZeroPadding2D((1, 1))(selected_feats)
	conv_class1 = Convolution2D(256, 3, 3, activation='relu', name="class_1")(conv_class1)
	conv_class1 = MaxPooling2D(pool_size=(2, 2))(conv_class1)

	conv_class2 = ZeroPadding2D((1, 1))(conv_class1)
	conv_class2 = Convolution2D(256, 3, 3, activation='relu', name="class_2")(conv_class2)
	conv_class2 = MaxPooling2D(pool_size=(2, 2))(conv_class2)

	conv_class3 = ZeroPadding2D((1, 1))(conv_class2)
	conv_class3 = Convolution2D(256, 3, 3, activation='relu', name="class_3")(conv_class3)
	conv_class3 = MaxPooling2D(pool_size=(2, 2))(conv_class3)

	conv_class4 = ZeroPadding2D((1, 1))(conv_class3)
	conv_class4 = Convolution2D(256, 3, 3, activation='relu', name="class_4")(conv_class4)
	conv_class4 = MaxPooling2D(pool_size=(2, 2))(conv_class4)

	fcn_class = Flatten()(conv_class4)
	fcn_class = Dense(256, activation='relu', name="class_dense_1")(fcn_class)
	fcn_class = Dropout(.5)(fcn_class)
	fcn_class = Dense(256, activation='relu', name="class_dense_2")(fcn_class)
	class_vec = Dense(8, activation='softmax', name="class")(fcn_class)

	return Model(input=imgs,output=[pred_mat,class_vec])

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	mask_update = UpdateFeatureMask(fish_feats)

	model = ClassifierContainer(name,construct(),32,"adam", callbacks=[mask_update])
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
