import numpy as np

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import SGD

from fish.model_container import ModelContainer
from preprocess import FISH_CLASSES,ROWS,COLS,CHANNELS

inputs = Input((ROWS, COLS, CHANNELS))
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

conv9 = Dropout(.5)(conv9)

conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

if __name__ == '__main__':
    model = ModelContainer('fish_detector_test',model,lambda x: x,sgd,"localizer")
    model.train(nb_epoch=140)