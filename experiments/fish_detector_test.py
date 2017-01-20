from keras.models import Sequential
from keras.layers import Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense
from keras.optimizers import SGD
from keras import backend as K

from fish.detector_container import ModelContainer

model = Sequential()

model.add(ZeroPadding2D((1, 1), dim_ordering='tf', input_shape=(256, 256, 3)))
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

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

if __name__ == '__main__':
	model = ModelContainer('fish_detector_test',model,sgd,debug=0)
	model.isfish_train(n=256, nb_epoch=1000, samples_per_epoch=10000, nb_val_samples=2000)
