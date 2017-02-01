from keras.models import Sequential
from keras.layers import Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, BatchNormalization
from keras import backend as K

from fish.detector_container import ModelContainer

def construct(n):
	model = Sequential()

	model.add(BatchNormalization(input_shape=(n, n, 3)))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	return model

if __name__ == '__main__':
	import sys

	if len(sys.argv) > 1: # very very basic
		model = ModelContainer('detect_batch_adagrad',construct(int(sys.argv[1])),int(sys.argv[1]),"adagrad")
		model.isfish_train(nb_epoch=int(sys.argv[2]), batch_size=int(sys.argv[3]), samples_per_epoch=int(sys.argv[4]))
	else:
		model = ModelContainer('detect_batch_adagrad',construct(256),256,"adagrad")
		model.isfish_train(nb_epoch=100, batch_size=500, samples_per_epoch=1000)