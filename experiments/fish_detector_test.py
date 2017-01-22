from keras.models import Sequential
from keras.layers import Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, BatchNormalization
from keras.optimizers import SGD
from keras import backend as K

from fish.detector_container import ModelContainer

def construct(n):
	model = Sequential()

	model.add(BatchNormalization(axis=3, input_shape=(n, n, 3)))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='softmax'))

	return model

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

if __name__ == '__main__':
	import sys

	if len(sys.argv) > 1: # very very basic
		model = ModelContainer('fish_detector_test',construct(int(sys.argv[1])),sgd)
		model.isfish_train(n=int(sys.argv[1]), nb_epoch=int(sys.argv[2]), batch_size=int(sys.argv[3]), samples_per_epoch=int(sys.argv[4]))
	else:
		model = ModelContainer('fish_detector_test',construct(256),sgd)
		model.isfish_train(n=256, nb_epoch=100, batch_size=500, samples_per_epoch=1000)
