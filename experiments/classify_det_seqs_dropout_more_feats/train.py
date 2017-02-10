from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, BatchNormalization, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.classify import ClassifierContainer

def construct(n):
	input_chunks = Input(shape=(None,n,n,3))
	chunks = BatchNormalization()(input_chunks)

	input_locations = Input(shape=(None,2))
	locations = BatchNormalization()(input_locations)

	# Glimpse net. Architecture inspired by DRAM paper.
	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(32, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(64, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((1, 1)))(chunks)
	chunks = TimeDistributed(Convolution2D(128, 3, 3, activation='relu'))(chunks)
	chunks = TimeDistributed(Convolution2D(64, 1, 1, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	flattened_chunks = TimeDistributed(Flatten())(chunks)
	flattened_chunks = Masking()(flattened_chunks)
	feature_vectors = TimeDistributed(Dense(128,activation='relu'))(flattened_chunks)

	# Location encoder
	location_vectors = TimeDistributed(Dense(128,activation='relu'))(locations)
	location_vectors = Masking()(location_vectors)
	
	# Multiplicative where-what interaction
	hadamard = merge([location_vectors, feature_vectors], mode='mul')

	# Combine the feature-location sequences
	rnn = LSTM(256, dropout_W=.5, dropout_U=.5)(hadamard)

	# Predict class
	fcn = Dropout(.5)(rnn)
	fcn = Dense(256,activation='relu')(fcn)
	fcn = Dropout(.5)(fcn)
	fcn = Dense(128,activation='relu')(fcn)
	fcn = Dense(8,activation='softmax')(fcn)

	return Model(input=[input_chunks,input_locations],output=fcn)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(256),256,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
