import numpy as np
import random
import pandas as pd

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
from matplotlib import ticker

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

import tensorflow as tf
tf.python.control_flow_ops = tf

from fish.model_container import ModelContainer
from preprocess import FISH_CLASSES,ROWS,COLS,CHANNELS

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf',input_shape=(ROWS, COLS, CHANNELS)))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(FISH_CLASSES)))
model.add(Activation('sigmoid'))

if __name__ == '__main__':
	model = ModelContainer('test',model,lambda x: x)
	model.train()
	model.evaluate('test')
