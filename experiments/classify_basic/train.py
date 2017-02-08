from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, BatchNormalization, ZeroPadding2D, Convolution2D
from keras.layers.core import Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.classifier import ClassifierContainer

def construct(n):
	input_chunks = Input(shape=(None,n,n,3))
	chunks = BatchNormalization()(input_chunks)

	input_locations = Input(shape=(None,2))
	locations = BatchNormalization()(input_locations)

	# Glimpse net. Inspired by DRAM paper.
	chunks = TimeDistributed(Convolution2D(32, 5, 5, activation='relu'))(chunks)

	flattened_chunks = TimeDistributed(Flatten())(chunks)
	flattened_chunks = Masking()(flattened_chunks)
	feature_vectors = TimeDistributed(Dense(32,activation='relu'))(flattened_chunks)

	# Location encoder
	location_vectors = TimeDistributed(Dense(32,activation='relu'))(locations)
	location_vectors = Masking()(location_vectors)
	
	# Multiplicative where-what interaction
	hadamard = merge([location_vectors, feature_vectors], mode='mul')

	# Combine the feature-location sequences
	rnn = LSTM(32, activation='relu')(hadamard)

	# Predict class
	fcn = Dense(8,activation='softmax')(rnn)

	return Model(input=[input_chunks,input_locations],output=fcn)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(256),256,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
