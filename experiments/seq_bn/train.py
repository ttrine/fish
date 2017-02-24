import theano

from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.layers import SequenceBatchNormalization, GradientReversalLayer
from fish.classify import ClassifierContainer

def construct(n):
	input_chunks = Input(shape=(None,n,n,3))
	chunks = SequenceBatchNormalization()(input_chunks)

	input_locations = Input(shape=(None,2))
	locations = SequenceBatchNormalization()(input_locations)

	# Glimpse net. Architecture inspired by DRAM paper.
	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(16, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(32, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((1, 1)))(chunks)
	chunks = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	flattened_chunks = TimeDistributed(Flatten())(chunks)
	flattened_chunks = Masking()(flattened_chunks)
	feature_vectors = TimeDistributed(Dense(128,activation='relu'))(flattened_chunks)

	# Location encoder
	location_vectors = TimeDistributed(Dense(128,activation='relu'))(locations)
	location_vectors = Masking()(location_vectors)
	
	# Multiplicative where-what interaction
	hadamard_1 = merge([location_vectors, feature_vectors], mode='mul')

	# Combine the feature-location sequences and predict coverage sequence
	detect_rnn = LSTM(128, return_sequences=True, activation='sigmoid', consume_less="gpu")(hadamard_1)
	detect_rnn_nograd = TimeDistributed(GradientReversalLayer(0))(detect_rnn)
	detect_fcn = TimeDistributed(Dense(64,activation='relu'))(detect_rnn)
	cov_pr = TimeDistributed(Dense(1,activation='sigmoid'),name="coverage")(detect_fcn)

	hadamard_2 = merge([detect_rnn_nograd, hadamard_1], mode='mul')

	# Gate according to output of detect RNN and predict class
	classify_rnn = LSTM(128, consume_less="gpu")(hadamard_2)
	classify_fcn = Dense(64,activation='relu')(classify_rnn)
	class_pr = Dense(8,activation='softmax',name="class")(classify_fcn)

	return Model(input=[input_chunks,input_locations],output=[cov_pr,class_pr])

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(128),128,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
