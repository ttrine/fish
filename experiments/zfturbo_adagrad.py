import numpy as np

from keras.models import Sequential
from keras.layers import Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense
from keras.optimizers import Adagrad
from keras import backend as K

from fish.model_container import ModelContainer
from preprocess import FISH_CLASSES,ROWS,COLS,CHANNELS

model = Sequential()

model.add(ZeroPadding2D((1, 1), dim_ordering='tf', input_shape=(ROWS, COLS, CHANNELS)))
model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

adagrad = Adagrad()

if __name__ == '__main__':
    model = ModelContainer('zfturbo_adam',model,lambda x: x,adagrad)
    model.train()
